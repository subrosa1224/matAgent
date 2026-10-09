"""Focused tests for the Intern agent adapter."""

import json
from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from materials_screening.agent.errors import AgentModelError
from materials_screening.agent.intern_model import (
    InternAgentModel,
    _forced_literature_search_tool,
    _looks_like_schema_description,
)
from materials_screening.agent.model_base import MaterialAgentRequest
from materials_screening.agent.models import (
    AgentFinalDraft,
    AgentFunctionOutputItem,
    AgentMessageItem,
)
from materials_screening.agent.settings import AgentSettings
from materials_screening.agent.tool_base import (
    AgentToolDefinition,
    ToolSideEffect,
)


def _request(*, allow_tool_calls: bool) -> MaterialAgentRequest:
    return MaterialAgentRequest(
        instructions="You are a materials assistant.",
        input_items=(),
        tool_definitions=(),
        final_draft_schema=AgentFinalDraft.model_json_schema(),
        allow_tool_calls=allow_tool_calls,
    )


def test_final_json_schema_is_required_even_when_tools_are_allowed() -> None:
    messages = InternAgentModel._messages(_request(allow_tool_calls=True))
    final_instruction = messages[-1]["content"]
    assert "only one valid JSON object" in final_instruction
    assert '"status"' in final_instruction
    assert '"evidence_ids"' in final_instruction
    assert "same language as the most recent user message" in final_instruction
    assert "maxLength" not in final_instruction
    assert '"$defs"' not in final_instruction


def test_internal_schema_description_is_detected() -> None:
    leaked = (
        "用户提供了AgentFinalDraft对象的JSON模式定义，其中包括"
        "referenced_material_ids和follow_up_question，并设置maxLength。"
    )
    assert _looks_like_schema_description(leaked) is True
    assert _looks_like_schema_description("检索到8篇相关论文。") is False


def _literature_search_request(message: str) -> MaterialAgentRequest:
    tool = AgentToolDefinition(
        name="literature_search",
        description="search papers",
        parameters={
            "type": "object",
            "additionalProperties": False,
            "properties": {},
        },
        side_effect=ToolSideEffect.READ_ONLY,
        version="1",
    )
    return MaterialAgentRequest(
        instructions="literature",
        input_items=(AgentMessageItem(role="user", content=message),),
        tool_definitions=(tool,),
        final_draft_schema=AgentFinalDraft.model_json_schema(),
    )


def test_explicit_literature_discovery_forces_search_tool() -> None:
    request = _literature_search_request(
        "帮我检索3D打印生物活性玻璃支架孔结构与成骨的近年论文"
    )

    assert _forced_literature_search_tool(request) == "literature_search"


def test_candidate_pool_prescreen_forces_dedicated_tool() -> None:
    base = _literature_search_request("执行候选池文献预检：ZnO；Ga2O3；最终前5名")
    screen_tool = AgentToolDefinition(
        name="screen_candidate_literature",
        description="screen candidates",
        parameters={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        side_effect=ToolSideEffect.READ_ONLY,
        version="1",
    )
    request = MaterialAgentRequest(
        instructions=base.instructions,
        input_items=base.input_items,
        tool_definitions=(*base.tool_definitions, screen_tool),
        final_draft_schema=base.final_draft_schema,
    )

    assert _forced_literature_search_tool(request) == "screen_candidate_literature"


def test_candidate_screen_answer_contains_download_boundary() -> None:
    answer = InternAgentModel._candidate_screen_answer(
        {
            "screening_id": "lit-screen-abc",
            "candidate_count": 50,
            "qualifying_candidate_count": 1,
            "finalists": [
                {
                    "formula": "ZnO",
                    "original_rank": 12,
                    "evidence_grade": "A",
                    "matching_paper_count": 2,
                    "note": "有器件证据",
                }
            ],
            "download_candidates": [
                {
                    "title": "ZnO UV detector",
                    "doi": "10.5/example",
                    "application_evidence_grade": "A",
                    "landing_page_url": "https://example.test/paper",
                }
            ],
        }
    )

    assert "计划 50 种材料" in answer
    assert "原属性排名" in answer
    assert "10.5/example" in answer
    assert "完整候选与检索状态保留在预检快照" in answer
    assert "上传 PDF" in answer
    assert "不是全文确认" in answer


@pytest.mark.parametrize(
    "message",
    [
        "请预览我上传的论文 PDF",
        "深度分析第1篇论文",
        "综合这些论文并生成主题报告",
    ],
)
def test_pdf_workflows_do_not_force_topic_search(message: str) -> None:
    assert _forced_literature_search_tool(_literature_search_request(message)) is None


def test_search_is_not_forced_again_after_tool_output() -> None:
    base = _literature_search_request("检索近年论文")
    request = MaterialAgentRequest(
        instructions=base.instructions,
        input_items=(
            *base.input_items,
            AgentFunctionOutputItem(call_id="search-1", output='{"status":"ok"}'),
        ),
        tool_definitions=base.tool_definitions,
        final_draft_schema=base.final_draft_schema,
    )

    assert _forced_literature_search_tool(request) is None


@pytest.mark.parametrize(
    "tool_name", ["literature_search", "openalex_search", "s2_search"]
)
def test_literature_search_answer_is_compacted_before_validation(
    tool_name: str,
) -> None:
    papers = [
        {
            "title": f"Paper {index}",
            "year": 2025,
            "doi": f"10.1/{index}",
            "authors": ["Author A", "Author B"],
            "venue": "Journal",
            "abstract": "A" * 1000,
            "selection_reason": "主题、材料和性能均匹配",
        }
        for index in range(30)
    ]
    output = AgentFunctionOutputItem(
        call_id="lit-call",
        output=json.dumps(
            {
                "status": "ok",
                "tool_name": tool_name,
                "evidence_id": "evidence-lit",
                "output": {
                    "papers": papers,
                    "returned_count": 30,
                    "warnings": [],
                },
            }
        ),
    )
    request = MaterialAgentRequest(
        instructions="literature",
        input_items=(
            AgentMessageItem(role="user", content="检索近年论文"),
            output,
        ),
        tool_definitions=(),
        final_draft_schema=AgentFinalDraft.model_json_schema(),
    )
    draft = AgentFinalDraft(status="completed", answer="x" * 8000)
    grounded = InternAgentModel._ground_final_draft(request, draft)
    assert "找到 30 篇" in grounded.answer
    assert "Paper 14" in grounded.answer
    assert "Paper 15" not in grounded.answer
    assert "重点论文详情（第1–5篇）" in grounded.answer
    assert len(grounded.answer) < 8000


def test_literature_tool_result_builds_deterministic_final_draft() -> None:
    output = AgentFunctionOutputItem(
        call_id="lit-call",
        output=json.dumps(
            {
                "status": "ok",
                "tool_name": "literature_search",
                "evidence_id": "evidence-lit",
                "output": {
                    "papers": [
                        {
                            "title": "A real provider paper",
                            "year": 2025,
                            "doi": "10.1/provider",
                            "authors": ["Author A", "Author B"],
                            "venue": "Acta Biomaterialia",
                            "cited_by_count": 12,
                            "access_status": "abstract_available",
                            "selection_reason": "匹配孔结构与成骨主题",
                            "abstract": "Provider supplied abstract text.",
                        }
                    ],
                    "returned_count": 1,
                    "query_id": "lit-query-1",
                    "expanded_query": {
                        "search_queries": ["bioactive glass pore osteogenesis"]
                    },
                    "provider_statuses": [
                        {
                            "provider": "openalex",
                            "status": "ok",
                            "records_returned": 1,
                        }
                    ],
                    "warnings": [],
                },
            }
        ),
    )
    request = MaterialAgentRequest(
        instructions="literature",
        input_items=(
            AgentMessageItem(role="user", content="检索近年论文"),
            output,
        ),
        tool_definitions=(),
        final_draft_schema=AgentFinalDraft.model_json_schema(),
        allow_tool_calls=False,
    )

    draft = InternAgentModel._literature_result_draft(request)

    assert draft is not None
    assert "A real provider paper" in draft.answer
    assert "10.1/provider" in draft.answer
    assert "lit-query-1" in draft.answer
    assert "Acta Biomaterialia" in draft.answer
    assert "Provider supplied abstract text." in draft.answer
    assert "openalex：正常" in draft.answer
    assert draft.evidence_ids == ["evidence-lit"]


def test_literature_follow_up_reuses_history_for_next_detail_page() -> None:
    papers = [
        {
            "title": f"Provider Paper {index}",
            "year": 2025,
            "doi": f"10.1/{index}",
            "authors": [f"Author {index}"],
            "abstract": f"Abstract {index}",
        }
        for index in range(1, 16)
    ]
    output = AgentFunctionOutputItem(
        call_id="lit-call",
        output=json.dumps(
            {
                "status": "ok",
                "tool_name": "literature_search",
                "evidence_id": "evidence-lit",
                "output": {
                    "query_id": "lit-query-1",
                    "papers": papers,
                    "returned_count": 15,
                    "warnings": [],
                },
            }
        ),
    )
    request = MaterialAgentRequest(
        instructions="literature",
        input_items=(
            AgentMessageItem(role="user", content="检索相关论文"),
            output,
            AgentMessageItem(role="assistant", content="previous result"),
            AgentMessageItem(role="user", content="我还想要后面几篇的详细信息"),
        ),
        tool_definitions=(),
        final_draft_schema=AgentFinalDraft.model_json_schema(),
        allow_tool_calls=True,
    )

    draft = InternAgentModel._literature_result_draft(request)

    assert draft is not None
    assert "第6–10篇" in draft.answer
    assert "Provider Paper 6" in draft.answer
    assert "Provider Paper 10" in draft.answer
    assert "Provider Paper 5" not in draft.answer
    assert "Provider Paper 11" not in draft.answer
    assert "实际检索式" not in draft.answer
    assert draft.evidence_ids == ["evidence-lit"]


def test_literature_follow_up_can_expand_all_remaining_candidates() -> None:
    papers = [
        {
            "title": f"Provider Paper {index}",
            "year": 2025,
            "doi": f"10.1/{index}",
            "authors": [f"Author {index}"],
            "abstract": "A" * 1000,
        }
        for index in range(1, 16)
    ]
    output = AgentFunctionOutputItem(
        call_id="lit-call",
        output=json.dumps(
            {
                "status": "ok",
                "tool_name": "literature_search",
                "evidence_id": "evidence-lit",
                "output": {
                    "query_id": "lit-query-1",
                    "papers": papers,
                    "returned_count": 15,
                    "warnings": [],
                },
            }
        ),
    )
    request = MaterialAgentRequest(
        instructions="literature",
        input_items=(
            AgentMessageItem(role="user", content="检索相关论文"),
            output,
            AgentMessageItem(role="assistant", content="previous result"),
            AgentMessageItem(role="user", content="展开后面所有篇章的详细信息"),
        ),
        tool_definitions=(),
        final_draft_schema=AgentFinalDraft.model_json_schema(),
    )

    draft = InternAgentModel._literature_result_draft(request)

    assert draft is not None
    assert "第6–15篇" in draft.answer
    assert "Provider Paper 6" in draft.answer
    assert "Provider Paper 15" in draft.answer
    assert "Provider Paper 5" not in draft.answer
    assert len(draft.answer) < 8000


def test_tools_disabled_turn_explicitly_requires_final_json() -> None:
    messages = InternAgentModel._messages(_request(allow_tool_calls=False))
    assert len(messages) == 1
    assert "Function calls are disabled" in messages[0]["content"]
    assert "only one valid JSON object" in messages[0]["content"]


def test_array_tool_arguments_are_normalized_from_intern_strings() -> None:
    tool = AgentToolDefinition(
        name="get_material_details",
        description="details",
        parameters={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "material_ids": {"type": "array", "items": {"type": "string"}},
                "fields": {"type": "array", "items": {"type": "string"}},
            },
        },
        side_effect=ToolSideEffect.READ_ONLY,
        version="1",
    )
    request = MaterialAgentRequest(
        instructions="materials",
        input_items=(),
        tool_definitions=(tool,),
        final_draft_schema=AgentFinalDraft.model_json_schema(),
    )

    normalized = InternAgentModel._normalize_tool_arguments(
        request,
        "get_material_details",
        '{"material_ids":"mp-149","fields":"formula_pretty, elements"}',
    )

    assert json.loads(normalized) == {
        "material_ids": ["mp-149"],
        "fields": ["formula_pretty", "elements", "material_id"],
    }


def test_non_array_tool_arguments_are_not_coerced() -> None:
    tool = AgentToolDefinition(
        name="lookup",
        description="lookup",
        parameters={
            "type": "object",
            "additionalProperties": False,
            "properties": {"query_id": {"type": "string"}},
        },
        side_effect=ToolSideEffect.READ_ONLY,
        version="1",
    )
    request = MaterialAgentRequest(
        instructions="materials",
        input_items=(),
        tool_definitions=(tool,),
        final_draft_schema=AgentFinalDraft.model_json_schema(),
    )
    raw = '{"query_id":"query-1"}'

    assert InternAgentModel._normalize_tool_arguments(request, "lookup", raw) == raw


@pytest.mark.parametrize(
    ("user_text", "keeps_strict_stability"),
    [
        (
            "筛选仅含 Li、Fe、O，凸包上能量不超过 0.05 的稳定材料",
            False,
        ),
        (
            "筛选仅含 Li、Fe、O，凸包上能量不超过 0.05 的严格稳定材料",
            True,
        ),
    ],
)
def test_exact_element_scope_and_stability_threshold_are_normalized(
    user_text: str, keeps_strict_stability: bool
) -> None:
    tool = AgentToolDefinition(
        name="search_materials",
        description="search",
        parameters={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "required_elements": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "chemsys": {"type": ["string", "null"]},
                "filters": {"type": "array", "items": {"type": "object"}},
            },
        },
        side_effect=ToolSideEffect.READ_ONLY,
        version="1",
    )
    request = MaterialAgentRequest(
        instructions="materials",
        input_items=(AgentMessageItem(role="user", content=user_text),),
        tool_definitions=(tool,),
        final_draft_schema=AgentFinalDraft.model_json_schema(),
    )
    raw = json.dumps(
        {
            "required_elements": ["Li", "Fe", "O"],
            "filters": [
                {
                    "field": "energy_above_hull_ev_atom",
                    "operator": "lte",
                    "value": 0.05,
                },
                {
                    "field": "is_stable",
                    "operator": "eq",
                    "value": True,
                },
            ],
        }
    )

    normalized = json.loads(
        InternAgentModel._normalize_tool_arguments(request, "search_materials", raw)
    )

    assert normalized["chemsys"] == "Li-Fe-O"
    assert normalized["required_elements"] == []
    has_stable = any(item["field"] == "is_stable" for item in normalized["filters"])
    assert has_stable is keeps_strict_stability


def test_explicit_material_detail_fields_are_added_to_search() -> None:
    tool = AgentToolDefinition(
        name="search_materials",
        description="search",
        parameters={
            "type": "object",
            "additionalProperties": False,
            "properties": {"fields": {"type": "array", "items": {"type": "string"}}},
        },
        side_effect=ToolSideEffect.READ_ONLY,
        version="1",
    )
    request = MaterialAgentRequest(
        instructions="materials",
        input_items=(
            AgentMessageItem(
                role="user",
                content=(
                    "查询Fe2O3的材料详情，返回化学式、元素、带隙、密度、"
                    "形成能、凸包上能量、稳定性、晶系和空间群"
                ),
            ),
        ),
        tool_definitions=(tool,),
        final_draft_schema=AgentFinalDraft.model_json_schema(),
    )

    normalized = json.loads(
        InternAgentModel._normalize_tool_arguments(
            request,
            "search_materials",
            '{"fields":["material_id","formula_pretty"]}',
        )
    )

    assert normalized["fields"] == [
        "material_id",
        "formula_pretty",
        "elements",
        "band_gap_ev",
        "density_g_cm3",
        "formation_energy_ev_atom",
        "energy_above_hull_ev_atom",
        "is_stable",
        "crystal_system",
        "spacegroup_symbol",
        "spacegroup_number",
    ]


def test_system_message_never_appears_after_user_history() -> None:
    request = MaterialAgentRequest(
        instructions="You are a materials assistant.",
        input_items=(AgentMessageItem(role="user", content="search materials"),),
        tool_definitions=(),
        final_draft_schema=AgentFinalDraft.model_json_schema(),
        allow_tool_calls=True,
    )
    messages = InternAgentModel._messages(request)
    assert [message["role"] for message in messages] == ["system", "user"]
    assert "final-answer schema" in messages[0]["content"]
    assert "same language as the most recent user message" in messages[0]["content"]


def test_plain_text_final_answer_is_repaired_to_json() -> None:
    calls: list[dict[str, object]] = []
    responses = [
        SimpleNamespace(
            id="first",
            model="intern-s2-preview-35b",
            usage=None,
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content="查询已经完成。", tool_calls=[]),
                )
            ],
        ),
        SimpleNamespace(
            id="repair",
            model="intern-s2-preview-35b",
            usage=None,
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content=(
                            '{"status":"completed","answer":"查询已经完成。",'
                            '"active_workflow_thread_id":null,'
                            '"referenced_material_ids":[],"evidence_ids":[],'
                            '"warnings":[],"follow_up_question":null}'
                        ),
                        tool_calls=[],
                    ),
                )
            ],
        ),
    ]

    class _Completions:
        def create(self, **kwargs: object) -> object:
            calls.append(kwargs)
            return responses.pop(0)

    client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))
    model = InternAgentModel(
        AgentSettings(), client=client, api_key=SecretStr("test-token")
    )
    response = model.generate(_request(allow_tool_calls=True))

    assert response.message_text is not None
    assert (
        AgentFinalDraft.model_validate_json(response.message_text).answer
        == "查询已经完成。"
    )
    assert len(calls) == 2
    assert "tool_choice" not in calls[0]
    assert "tool_choice" not in calls[1]
    assert "tools" not in calls[1]
    assert calls[1]["extra_body"] == {"thinking_mode": False}


def test_incomplete_json_final_answer_is_repaired_to_agent_draft() -> None:
    calls: list[dict[str, object]] = []
    responses = [
        SimpleNamespace(
            id="first",
            model="intern-s2-preview-35b",
            usage=None,
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content='{"answer":"好的，我将继续处理。"}',
                        tool_calls=[],
                    ),
                )
            ],
        ),
        SimpleNamespace(
            id="repair",
            model="intern-s2-preview-35b",
            usage=None,
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content=(
                            '{"status":"completed",'
                            '"answer":"好的，我将继续处理。",'
                            '"active_workflow_thread_id":null,'
                            '"referenced_material_ids":[],"evidence_ids":[],'
                            '"warnings":[],"follow_up_question":null}'
                        ),
                        tool_calls=[],
                    ),
                )
            ],
        ),
    ]

    class _Completions:
        def create(self, **kwargs: object) -> object:
            calls.append(kwargs)
            return responses.pop(0)

    client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))
    model = InternAgentModel(
        AgentSettings(), client=client, api_key=SecretStr("test-token")
    )

    response = model.generate(_request(allow_tool_calls=False))

    draft = AgentFinalDraft.model_validate_json(response.message_text or "")
    assert draft.answer == "好的，我将继续处理。"
    assert len(calls) == 2
    assert "tools" not in calls[1]


def test_invalid_repair_falls_back_to_local_protocol_envelope() -> None:
    responses = [
        SimpleNamespace(
            id="first",
            model="intern-s2-preview-35b",
            usage=None,
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content='{"answer":"筛选完成，共 20 条。"}',
                        tool_calls=[],
                    ),
                )
            ],
        ),
        SimpleNamespace(
            id="repair",
            model="intern-s2-preview-35b",
            usage=None,
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content='{"status":"done","answer":"筛选完成，共 20 条。"}',
                        tool_calls=[],
                    ),
                )
            ],
        ),
    ]

    class _Completions:
        def create(self, **kwargs: object) -> object:
            return responses.pop(0)

    client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))
    model = InternAgentModel(
        AgentSettings(), client=client, api_key=SecretStr("test-token")
    )

    response = model.generate(_request(allow_tool_calls=False))

    draft = AgentFinalDraft.model_validate_json(response.message_text or "")
    assert draft.status == "completed"
    assert draft.answer == "筛选完成，共 20 条。"
    assert draft.evidence_ids == []


def test_empty_final_message_uses_successful_material_tool_evidence() -> None:
    response = SimpleNamespace(
        id="empty-final",
        model="intern-s2-preview-35b",
        usage=None,
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content="", tool_calls=[]),
            )
        ],
    )

    class _Completions:
        def create(self, **kwargs: object) -> object:
            return response

    request = MaterialAgentRequest(
        instructions="You are a materials assistant.",
        input_items=(
            AgentMessageItem(
                role="user",
                content="筛选材料，返回材料 ID、化学式和带隙",
            ),
            AgentFunctionOutputItem(
                call_id="call-1",
                output=json.dumps(
                    {
                        "status": "ok",
                        "tool_name": "search_materials",
                        "evidence_id": "ev-1",
                        "output": {
                            "matched_count": 1,
                            "fields": [
                                "material_id",
                                "formula_pretty",
                                "band_gap_ev",
                            ],
                            "materials": [
                                {
                                    "material_id": "mp-19017",
                                    "formula_pretty": "LiFePO4",
                                    "band_gap_ev": 3.9224,
                                }
                            ],
                        },
                    },
                    ensure_ascii=False,
                ),
            ),
        ),
        tool_definitions=(),
        final_draft_schema=AgentFinalDraft.model_json_schema(),
        allow_tool_calls=False,
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))
    model = InternAgentModel(
        AgentSettings(), client=client, api_key=SecretStr("test-token")
    )

    result = model.generate(request)

    draft = AgentFinalDraft.model_validate_json(result.message_text or "")
    assert result.status.value == "completed"
    assert "| mp-19017 | LiFePO4 | 3.9224 |" in draft.answer
    assert draft.evidence_ids == ["ev-1"]


def test_empty_final_message_without_row_evidence_still_fails() -> None:
    response = SimpleNamespace(
        id="empty-final",
        model="intern-s2-preview-35b",
        usage=None,
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content="", tool_calls=[]),
            )
        ],
    )

    class _Completions:
        def create(self, **kwargs: object) -> object:
            return response

    client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))
    model = InternAgentModel(
        AgentSettings(), client=client, api_key=SecretStr("test-token")
    )

    with pytest.raises(AgentModelError, match="empty final message"):
        model.generate(_request(allow_tool_calls=False))


def test_generic_provider_error_is_not_accepted_as_final_answer() -> None:
    response = SimpleNamespace(
        id="error",
        model="intern-s2-preview-35b",
        usage=None,
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(
                    content=(
                        "An error occurred while processing your prompt. "
                        "Please check your input and try again."
                    ),
                    tool_calls=[],
                ),
            )
        ],
    )

    class _Completions:
        def create(self, **kwargs: object) -> object:
            return response

    client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))
    model = InternAgentModel(
        AgentSettings(), client=client, api_key=SecretStr("test-token")
    )

    with pytest.raises(AgentModelError, match="generic service error"):
        model.generate(_request(allow_tool_calls=True))


def test_chinese_user_gets_language_repair_without_evidence_changes() -> None:
    responses = [
        SimpleNamespace(
            id="first",
            model="intern-s2-preview-35b",
            usage=None,
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content=(
                            '{"status":"completed","answer":"Search completed.",'
                            '"active_workflow_thread_id":null,'
                            '"referenced_material_ids":["mp-149"],'
                            '"evidence_ids":["ev-1"],"warnings":[],'
                            '"follow_up_question":null}'
                        ),
                        tool_calls=[],
                    ),
                )
            ],
        ),
        SimpleNamespace(
            id="language-mismatch",
            model="intern-s2-preview-35b",
            usage=None,
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content=(
                            '{"status":"completed","answer":"Search completed.",'
                            '"active_workflow_thread_id":null,'
                            '"referenced_material_ids":[],"evidence_ids":[],'
                            '"warnings":[],"follow_up_question":null}'
                        ),
                        tool_calls=[],
                    ),
                )
            ],
        ),
        SimpleNamespace(
            id="language",
            model="intern-s2-preview-35b",
            usage=None,
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content=(
                            '{"status":"completed","answer":"查询完成。",'
                            '"active_workflow_thread_id":null,'
                            '"referenced_material_ids":[],"evidence_ids":[],'
                            '"warnings":[],"follow_up_question":null}'
                        ),
                        tool_calls=[],
                    ),
                )
            ],
        ),
    ]

    class _Completions:
        def create(self, **kwargs: object) -> object:
            return responses.pop(0)

    client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))
    model = InternAgentModel(
        AgentSettings(), client=client, api_key=SecretStr("test-token")
    )
    request = MaterialAgentRequest(
        instructions="You are a materials assistant.",
        input_items=(AgentMessageItem(role="user", content="查询这个材料"),),
        tool_definitions=(),
        final_draft_schema=AgentFinalDraft.model_json_schema(),
        allow_tool_calls=False,
    )
    response = model.generate(request)
    draft = AgentFinalDraft.model_validate_json(response.message_text or "")

    assert draft.answer == "查询完成。"
    assert draft.referenced_material_ids == ["mp-149"]
    assert draft.evidence_ids == ["ev-1"]


def test_missing_grounding_metadata_is_filled_from_successful_tool_output() -> None:
    tool_output = AgentFunctionOutputItem(
        call_id="call-1",
        output=(
            '{"status":"ok","evidence_id":"ev-1","output":'
            '{"materials":[{"material_id":"mp-149"}]}}'
        ),
    )
    request = MaterialAgentRequest(
        instructions="You are a materials assistant.",
        input_items=(tool_output,),
        tool_definitions=(),
        final_draft_schema=AgentFinalDraft.model_json_schema(),
        allow_tool_calls=False,
    )
    draft = AgentFinalDraft(
        status="completed",
        answer="候选材料为 mp-149。",
    )

    grounded = InternAgentModel._ground_final_draft(request, draft)

    assert grounded.evidence_ids == ["ev-1"]
    assert grounded.referenced_material_ids == ["mp-149"]


def test_failed_tool_output_is_not_used_as_grounding() -> None:
    request = MaterialAgentRequest(
        instructions="You are a materials assistant.",
        input_items=(
            AgentFunctionOutputItem(
                call_id="call-1",
                output=(
                    '{"status":"error","evidence_id":"ev-bad",'
                    '"output":{"material_id":"mp-149"}}'
                ),
            ),
        ),
        tool_definitions=(),
        final_draft_schema=AgentFinalDraft.model_json_schema(),
        allow_tool_calls=False,
    )
    draft = AgentFinalDraft(status="completed", answer="候选材料为 mp-149。")

    assert InternAgentModel._ground_final_draft(request, draft) == draft


def test_requested_material_rows_are_restored_from_tool_evidence() -> None:
    request = MaterialAgentRequest(
        instructions="You are a materials assistant.",
        input_items=(
            AgentMessageItem(
                role="user",
                content="筛选稳定材料，返回材料 ID、化学式、带隙和凸包上能量",
            ),
            AgentFunctionOutputItem(
                call_id="call-1",
                output=json.dumps(
                    {
                        "status": "ok",
                        "tool_name": "search_materials",
                        "evidence_id": "ev-1",
                        "output": {
                            "fields": [
                                "material_id",
                                "formula_pretty",
                                "band_gap_ev",
                                "energy_above_hull_ev_atom",
                            ],
                            "materials": [
                                {
                                    "material_id": "mp-19017",
                                    "formula_pretty": "LiFePO4",
                                    "band_gap_ev": 3.9224,
                                    "energy_above_hull_ev_atom": 0.0,
                                }
                            ],
                        },
                    },
                    ensure_ascii=False,
                ),
            ),
        ),
        tool_definitions=(),
        final_draft_schema=AgentFinalDraft.model_json_schema(),
        allow_tool_calls=False,
    )
    draft = AgentFinalDraft(
        status="completed",
        answer="mp-19017 的带隙为 0.0 eV。",
        warnings=["该材料可能不满足筛选条件。"],
        follow_up_question="是否需要导出？",
    )

    grounded = InternAgentModel._ground_final_draft(request, draft)

    assert "| mp-19017 | LiFePO4 | 3.9224 | 0 |" in grounded.answer
    assert "0.0 eV" not in grounded.answer
    assert grounded.answer.startswith("查询完成：共匹配 1 条，当前展示 1 条。")
    assert grounded.referenced_material_ids == ["mp-19017"]
    assert grounded.evidence_ids == ["ev-1"]
    assert grounded.warnings == []
    assert grounded.follow_up_question is None


def test_material_rows_are_not_forced_for_count_only_question() -> None:
    request = MaterialAgentRequest(
        instructions="You are a materials assistant.",
        input_items=(
            AgentMessageItem(role="user", content="符合条件的材料有多少个？"),
            AgentFunctionOutputItem(
                call_id="call-1",
                output=(
                    '{"status":"ok","evidence_id":"ev-1","output":'
                    '{"fields":["material_id"],'
                    '"materials":[{"material_id":"mp-149"}]}}'
                ),
            ),
        ),
        tool_definitions=(),
        final_draft_schema=AgentFinalDraft.model_json_schema(),
        allow_tool_calls=False,
    )
    draft = AgentFinalDraft(status="completed", answer="共有 1 个。")

    grounded = InternAgentModel._ground_final_draft(request, draft)

    assert grounded.answer == "共有 1 个。"
    assert grounded.referenced_material_ids == []


def test_search_rows_are_not_appended_after_outlier_analysis() -> None:
    request = MaterialAgentRequest(
        instructions="You are a materials assistant.",
        input_items=(
            AgentMessageItem(role="user", content="列出带隙离群材料"),
            AgentFunctionOutputItem(
                call_id="search-1",
                output=(
                    '{"status":"ok","tool_name":"search_materials",'
                    '"evidence_id":"ev-search","output":'
                    '{"fields":["material_id"],'
                    '"materials":[{"material_id":"mp-149"}]}}'
                ),
            ),
            AgentFunctionOutputItem(
                call_id="outlier-1",
                output=(
                    '{"status":"ok","tool_name":"detect_material_outliers",'
                    '"evidence_id":"ev-outlier","output":'
                    '{"report":{"distribution":{"count":1},"records":[]}}}'
                ),
            ),
        ),
        tool_definitions=(),
        final_draft_schema=AgentFinalDraft.model_json_schema(),
        allow_tool_calls=False,
    )
    draft = AgentFinalDraft(
        status="completed",
        answer="IQR 检测样本数为 1，未发现离群材料。",
    )

    grounded = InternAgentModel._ground_final_draft(request, draft)

    assert grounded.answer == draft.answer
    assert grounded.referenced_material_ids == []
    assert grounded.evidence_ids == ["ev-search", "ev-outlier"]
