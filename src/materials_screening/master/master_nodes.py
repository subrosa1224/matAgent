"""Master agent graph nodes and pure routing functions.

Every node has one responsibility: prepare a turn, call the master model,
execute sub-agent delegations, validate the final draft, or finalize.
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from langgraph.runtime import Runtime
from pydantic import ValidationError

from materials_screening.agent.errors import AgentInvariantError, AgentModelError
from materials_screening.agent.model_base import (
    AgentModelStatus,
    MaterialAgentRequest,
)
from materials_screening.agent.models import (
    AgentFinalDraft,
    AgentFinalStatus,
    AgentFunctionCallItem,
    AgentFunctionOutputItem,
    AgentMessageItem,
)
from materials_screening.agent.tool_base import AgentToolDefinition, ToolSideEffect
from materials_screening.sub_agents.materials_database.formula_screening import (
    parse_formula_screening,
)
from materials_screening.sub_agents.materials_database.screening_handoff import (
    with_screening_handoff_scope,
)

from .application_feasibility import render_uv_candidate_queue
from .data_analysis_handoffs import DataAnalysisCrossAgentCoordinator
from .master_context import MasterAgentContext
from .master_state import MasterAgentState
from .sub_agent_spec import SubAgentCall

NODE_PREPARE_TURN = "prepare_turn"
NODE_CALL_MASTER_MODEL = "call_master_model"
NODE_EXECUTE_SUB_AGENT = "execute_sub_agent"
NODE_VALIDATE_FINAL = "validate_final"
NODE_FINALIZE_SUCCESS = "finalize_success"
NODE_FINALIZE_ERROR = "finalize_error"
NODE_FINALIZE_CANCELLED = "finalize_cancelled"

_CONVERSATION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,253}$")
_OUTLIER_REQUEST_MARKERS = (
    "outlier",
    "anomaly",
    "\u79bb\u7fa4",
    "\u5f02\u5e38\u68c0\u6d4b",
    "\u5f02\u5e38\u503c",
)
_MATERIAL_DATABASE_REQUEST_MARKERS = (
    "materials project",
    "material",
    "mp-",
    "材料",
    "氧化物",
    "化合物",
    "带隙",
    "密度",
    "形成能",
    "凸包",
    "稳定",
    "晶系",
    "元素",
    "筛选",
    "top-k",
    "top k",
    "排序",
    "比较",
    "统计",
    "导出",
)
_DATA_ANALYSIS_REQUEST_MARKERS = (
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
_LITERATURE_PROCESS_MARKERS = (
    "合成",
    "水热",
    "沉淀",
    "溶胶",
    "ph",
    "温度",
    "时间",
    "老化",
    "物相",
    "结晶度",
    "粒径",
    "颗粒",
    "形貌",
)
_LITERATURE_RELATION_MARKERS = ("影响", "关系", "如何", "作用", "趋势")
_LITERATURE_HANDOFF_PATTERN = re.compile(
    r"LITERATURE_DATASET_HANDOFF:\s*dataset_id="
    r"(dataset-[A-Za-z0-9][A-Za-z0-9._:-]{0,247})"
)
_MATERIAL_QUERY_HANDOFF_PATTERN = re.compile(
    r"MATERIAL_QUERY_HANDOFF:\s*query_id="
    r"(query-[A-Za-z0-9][A-Za-z0-9._:-]{0,247})"
)
_EXPLICIT_SOURCE_PATTERN = re.compile(
    r"(?:data[\\/].*?\.(?:csv|json)\b|\bmp-[A-Za-z0-9]+\b|"
    r"\b(?:workflow[ _-]?thread|thread[ _-]?id|thread)\b|"
    r"工作流(?:线程)?|\b[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b|"
    r"\b(?:[A-Z][a-z]?\d*){1,8}\b)",
    re.IGNORECASE,
)


def _has_explicit_material_source(message: str) -> bool:
    """Whether a follow-up already names a file, thread, id, or formula."""
    return _EXPLICIT_SOURCE_PATTERN.search(message) is not None


def _is_material_database_request(message: str) -> bool:
    """Recognize requests owned by the unified materials database agent."""
    normalized = message.lower()
    return any(
        marker in normalized
        for marker in (
            *_OUTLIER_REQUEST_MARKERS,
            *_MATERIAL_DATABASE_REQUEST_MARKERS,
        )
    )


def _is_data_analysis_request(message: str) -> bool:
    """Recognize explicit requests for registered tabular datasets."""

    normalized = message.lower()
    return any(marker in normalized for marker in _DATA_ANALYSIS_REQUEST_MARKERS)


def _is_literature_evidence_request(message: str) -> bool:
    """Recognize synthesis/process questions that need literature evidence."""

    normalized = message.casefold()
    return any(marker in normalized for marker in _LITERATURE_PROCESS_MARKERS) and any(
        marker in normalized for marker in _LITERATURE_RELATION_MARKERS
    )


def _screening_final_limit(message: str) -> int:
    match = re.search(r"(?:前\s*|top\s*)(\d{1,2})\s*(?:名|种)?", message, re.IGNORECASE)
    return max(1, min(20, int(match.group(1)))) if match else 5


def _is_uv_detector_application(message: str) -> bool:
    """Use detector-specific rules only for an explicit UV detector topic."""

    normalized = message.casefold()
    return (
        "紫外" in normalized
        or "ultraviolet" in normalized
        or re.search(r"\buv\b", normalized) is not None
    ) and any(
        marker in normalized
        for marker in ("光电探测", "探测器", "photodetector", "detector")
    )


def _requires_material_screening_chain(message: str) -> bool:
    """Recognize requested stages, not an application-evaluation vocabulary.

    A literature mention in background is insufficient. Remove explicit opt-out
    clauses before looking for affirmative actions; ordinary database-only
    requests must still retain their direct-result path.
    """

    normalized = re.sub(
        r"(?:不需要|不要|无需|不必|不用|不做)[^。；;，,\n]*",
        "",
        message.casefold(),
    )
    literature_requested = re.search(
        r"(?:检索|查找|搜索|搜集|结合|\b(?:search|retrieve|review)\b)"
        r"[^。；;\n]{0,160}(?:文献|论文|实验研究|literature|papers?)"
        r"|(?:文献|论文|literature)\s*(?:检索|搜索)",
        normalized,
    )
    return (
        any(
            marker in normalized
            for marker in ("筛选", "查询", "查找", "寻找", "screen")
        )
        and any(marker in normalized for marker in ("分析", "分布", "统计"))
        and literature_requested is not None
    )


_MASTER_SYSTEM_PROMPT = (
    "You are a master orchestrator for a materials research assistant. "
    "You have access to specialized sub-agents that perform specific tasks. "
    "Each sub-agent is exposed as a function: delegate_to_<name>. "
    "Your job:\n\n"
    "1. CLASSIFY the user's request and decide which sub-agent(s) to "
    "delegate to.\n"
    "2. DELEGATE: call the appropriate function with a specific, detailed "
    "task description that includes all relevant context from the user's "
    "message.\n"
    "3. WAIT for the sub-agent result, then decide: is the answer "
    "sufficient? If yes, synthesize. If another sub-agent is needed, "
    "delegate again.\n"
    "4. For conversational, meta, or clarification questions that do not "
    "require a sub-agent, answer directly.\n"
    "5. SYNTHESIZE: combine all sub-agent results into a coherent final "
    "answer.\n\n"
    "Rules:\n"
    "- Make exactly one delegation call per response.\n"
    "- After receiving a sub-agent result, either delegate to another "
    "sub-agent or produce your final answer.\n"
    "- Never fabricate information; answer only from sub-agent results.\n"
    "- Never expose internal reasoning, API keys, or raw sub-agent outputs.\n"
    "- Write answer, warnings, and follow_up_question in the same language as "
    "the user's most recent message. Keep JSON keys, material IDs, chemical "
    "formulas, field names, and units unchanged.\n"
    "- For simple conversational questions, greetings, or clarifications, "
    "answer directly without delegating.\n"
    "- Your final answer MUST be a single JSON object with keys: "
    "status, answer, referenced_material_ids, evidence_ids, warnings, "
    "follow_up_question.\n"
    "- Use status 'completed' for successful answers, "
    "'needs_user_input' to ask the user for clarification, 'error' for "
    "failures."
)

RouteAfterModel = Literal[
    "execute_sub_agent", "validate_final", "finalize_error", "finalize_cancelled"
]
RouteAfterSubAgent = Literal[
    "call_master_model", "finalize_error", "finalize_cancelled"
]
RouteAfterValidation = Literal["finalize_success", "finalize_error"]


# ── Node: prepare_turn ───────────────────────────────────────────────────────


def prepare_turn_node(
    state: MasterAgentState,
    runtime: Runtime[MasterAgentContext],
) -> dict[str, Any]:
    context = runtime.context
    node = NODE_PREPARE_TURN
    conversation_id = state.get("conversation_id", "")
    user_message = state.get("user_message", "")
    if not _CONVERSATION_ID_PATTERN.fullmatch(conversation_id):
        return _fail(
            state,
            context,
            node,
            "INVALID_CONVERSATION",
            "conversation_id has an unsafe format",
        )
    message_bytes = len(user_message.encode("utf-8"))
    max_input_bytes = context.settings.agent_max_input_bytes
    if message_bytes > max_input_bytes:
        return _fail(
            state,
            context,
            node,
            "INPUT_TOO_LARGE",
            f"user message exceeds {max_input_bytes} bytes",
        )
    existing_items = list(state.get("input_items", []))
    return {
        "user_turn_id": context.id_generator.new_id(),
        "input_items": [
            *existing_items,
            AgentMessageItem(role="user", content=user_message).model_dump(mode="json"),
        ],
        "status": "running",
        "cancelled": False,
        "current_node": node,
        "model_call_count": 0,
        "sub_agent_call_count": 0,
        "pending_tool_calls": [],
        "executed_call_ids": [],
        "evidence_ids": [],
        "sub_agent_results": [],
        "final_draft": None,
        "final_response": None,
        "error": None,
        "events": [
            _event(
                state,
                context,
                node=node,
                event_type="turn_started",
                status="running",
                message="user turn started",
                metrics={},
            )
        ],
    }


# ── Node: call_master_model ───────────────────────────────────────────────────


def call_master_model_node(
    state: MasterAgentState,
    runtime: Runtime[MasterAgentContext],
) -> dict[str, Any]:
    context = runtime.context
    node = NODE_CALL_MASTER_MODEL
    if state.get("error") is not None:
        return {}
    if _is_cancelled(context):
        return _cancel(state, context, node)
    screening_report = _material_screening_chain_passthrough(
        state,
        coordinator=context.data_analysis_coordinator,
    )
    if screening_report is not None:
        message_text = json.dumps(screening_report, ensure_ascii=False)
        items = list(state.get("input_items", []))
        items.append(
            AgentMessageItem(role="assistant", content=message_text).model_dump(
                mode="json"
            )
        )
        return {
            "current_node": node,
            "input_items": items,
            "pending_tool_calls": [],
            "final_draft": screening_report,
            "status": "running",
            "events": [
                _progress(
                    state,
                    context,
                    node=node,
                    event_type="material_screening_chain_combined",
                    message="Master 已汇总材料初筛、数据分析与文献核验结果",
                    metrics={"model_call": state.get("model_call_count", 0)},
                )
            ],
        }
    user_message = state.get("user_message", "")
    if _requires_material_screening_chain(user_message):
        query_id = _material_query_handoff_id(state)
        results = _current_turn_sub_agent_results(state)
        # An unsuccessful delegation still consumed a slot. Do not retry it
        # automatically: literature uses the database snapshot independently.
        analysis_attempted = any(
            result.get("sub_agent_name") == "data_analysis"
            and result.get("status") in {"ok", "error"}
            for result in results
        )
        literature_attempted = any(
            result.get("sub_agent_name") == "literature"
            and result.get("status") in {"ok", "error"}
            for result in results
        )
        within_budget = (
            state.get("sub_agent_call_count", 0)
            < context.settings.agent_max_tool_calls_per_turn
        )
        coordinator = context.data_analysis_coordinator
        if query_id is not None and not analysis_attempted and coordinator is not None:
            delegate = context.sub_agent_registry.resolve_by_delegate_function(
                "delegate_to_data_analysis"
            )
            if delegate is not None and within_budget:
                try:
                    handoff = coordinator.material_query_to_analysis(
                        query_id,
                        task="分析材料候选的稳定性、带隙与密度分布",
                    )
                except ValueError:
                    handoff = None
                if handoff is not None:
                    group_dimensions = []
                    if re.search(
                        r"晶系|crystal[_ ]system", user_message, re.IGNORECASE
                    ):
                        group_dimensions.append("crystal_system")
                    if re.search(r"化学式|formula", user_message, re.IGNORECASE):
                        group_dimensions.append("formula_pretty")
                    if (
                        query_id.startswith("query-cohort-")
                        and "formula_pretty" not in group_dimensions
                    ):
                        group_dimensions.append("formula_pretty")
                    if not group_dimensions:
                        group_dimensions = ["crystal_system"]
                    grouping = (
                        "group_by_each=" + ",".join(group_dimensions)
                        if len(group_dimensions) > 1
                        else "group_by=" + group_dimensions[0]
                    )
                    task = (
                        f"对材料数据库候选数据集 {handoff.partition.dataset_id} "
                        "做描述统计。columns=band_gap_ev,density_g_cm3,"
                        f"energy_above_hull_ev_atom {grouping}。"
                        "报告样本数、缺失值和分布，不推断器件性能。"
                    )
                    return _queue_sub_agent_delegation(
                        state,
                        context,
                        node=node,
                        delegate_name=delegate.delegate_function_name,
                        task=task,
                        event_type="material_query_dataset_delegated",
                        message="材料候选数据已交给数据分析 Agent",
                        metrics={
                            "query_id": query_id,
                            "dataset_id": handoff.partition.dataset_id,
                        },
                    )
        if (
            query_id is not None
            and analysis_attempted
            and not literature_attempted
            and coordinator is not None
        ):
            delegate = context.sub_agent_registry.resolve_by_delegate_function(
                "delegate_to_literature"
            )
            if delegate is not None and within_budget:
                if not _is_uv_detector_application(user_message):
                    try:
                        raw_candidates = coordinator.material_query_candidates(
                            query_id, limit=20, unique_formulas=True
                        )
                    except ValueError:
                        raw_candidates = ()
                    if raw_candidates:
                        formulas = list(
                            dict.fromkeys(
                                str(row["formula_pretty"]) for row in raw_candidates
                            )
                        )
                        task = (
                            "检索与以下原始科研问题及数据库候选材料相关的实验文献。"
                            f"\n原始科研问题：{user_message}"
                            f"\n候选化学式：{'；'.join(formulas)}。"
                            f"查询快照：{query_id}；完整快照先按化学式去重，"
                            "按各化学式最低凸包能升序取最多20种做检索队列，"
                            "这是有限检索样本，不是应用性能排名或最终推荐。"
                            "使用 literature_search 通用检索，"
                            "保持原问题的应用方向与条件；"
                            "列出实际检索到的题名、年份、DOI与来源，并说明相关性与缺口。"
                            "不预设其他应用的证据分级；"
                            "不得把配方级论文对应为特定 MP 物相。"
                            "题名/摘要只能作为线索，不能据此确认应用可行性或实验性能；"
                            "需要全文时明确说明需提供对应论文正文，不得假装已完成全文核验。"
                        )
                        return _queue_sub_agent_delegation(
                            state,
                            context,
                            node=node,
                            delegate_name=delegate.delegate_function_name,
                            task=task,
                            event_type="screening_candidates_delegated_to_literature",
                            message="候选材料与原始科研问题已交给文献 Agent 做通用检索",
                            metrics={
                                "query_id": query_id,
                                "candidate_count": len(formulas),
                            },
                        )
                    # A missing snapshot must not fall into detector screening.
                    candidates = ()
                else:
                    try:
                        candidates = coordinator.prioritized_material_query_candidates(
                            query_id, limit=50, unique_formulas=True
                        )
                    except ValueError:
                        candidates = ()
                if candidates:
                    final_limit = _screening_final_limit(user_message)
                    supplementary_id = _supplementary_query_handoff_id(state)
                    supplementary_candidates = ()
                    if supplementary_id:
                        try:
                            supplemental = (
                                coordinator.prioritized_material_query_candidates(
                                    supplementary_id, limit=50, unique_formulas=True
                                )
                            )
                            strict_formulas = {row.formula_pretty for row in candidates}
                            supplementary_candidates = tuple(
                                row
                                for row in supplemental
                                if row.formula_pretty not in strict_formulas
                            )
                        except ValueError:
                            pass
                    candidate_text = "；".join(
                        f"{index}:{row.formula_pretty}"
                        for index, row in enumerate(candidates, 1)
                    )
                    task = (
                        "执行候选池文献预检（该候选池不是最终推荐）：按原属性排名检查"
                        f"以下 {len(candidates)} 种材料用于紫外光电探测的文献证据："
                        f"{candidate_text}。按 A/B/C/NONE 分级并用证据等级重排，"
                        f"最终最多列出前{final_limit}名，不足则如实说明；"
                        "只有 A/B 级论文进入下载清单。A 级须有实际"
                        "紫外探测器和器件指标，B 级为实验光学或不完整器件证据。"
                        "不得用无关论文凑足名额，也不得把配方级论文说成特定 MP 物相。"
                    )
                    if supplementary_candidates:
                        supplementary_text = "；".join(
                            f"S{index}:{row.formula_pretty}"
                            for index, row in enumerate(supplementary_candidates, 1)
                        )
                        task += (
                            f"\n独立探索池：{supplementary_text}。探索池下调计算带隙下限并限定二元氧化物，"
                            "不是原严格条件达标名单；与严格池分别排名，绝不用探索池补足严格前几名。"
                            f"探索池快照 {supplementary_id}；严格池快照 {query_id}。"
                        )
                    return _queue_sub_agent_delegation(
                        state,
                        context,
                        node=node,
                        delegate_name=delegate.delegate_function_name,
                        task=task,
                        event_type="screening_candidates_delegated_to_literature",
                        message="应用候选池已交给文献 Agent 做证据预检与重排",
                        metrics={
                            "query_id": query_id,
                            "candidate_count": len(candidates),
                        },
                    )
        if results or not within_budget:
            # A failed/missing handoff or exhausted budget is not a reason to
            # ask the model for another delegation or discard successful work.
            partial = _material_screening_partial_report(
                state,
                call_limit=context.settings.agent_max_tool_calls_per_turn,
            )
            items = list(state.get("input_items", []))
            items.append(
                AgentMessageItem(
                    role="assistant", content=json.dumps(partial, ensure_ascii=False)
                ).model_dump(mode="json")
            )
            return {
                "current_node": node,
                "input_items": items,
                "pending_tool_calls": [],
                "final_draft": partial,
                "status": "running",
                "events": [
                    _progress(
                        state,
                        context,
                        node=node,
                        event_type="material_screening_chain_incomplete",
                        message="Master 已保留成功结果，并标明失败或未执行的步骤",
                        metrics={
                            "sub_agent_calls": state.get("sub_agent_call_count", 0),
                            "sub_agent_call_limit": (
                                context.settings.agent_max_tool_calls_per_turn
                            ),
                        },
                    )
                ],
            }
    combined = _literature_analysis_passthrough(state)
    if combined is not None:
        message_text = json.dumps(combined, ensure_ascii=False)
        items = list(state.get("input_items", []))
        items.append(
            AgentMessageItem(role="assistant", content=message_text).model_dump(
                mode="json"
            )
        )
        return {
            "current_node": node,
            "input_items": items,
            "pending_tool_calls": [],
            "final_draft": combined,
            "status": "running",
            "events": [
                _progress(
                    state,
                    context,
                    node=node,
                    event_type="literature_analysis_combined",
                    message="Master 已汇总文献证据与数据分析结果",
                    metrics={"model_call": state.get("model_call_count", 0)},
                )
            ],
        }
    handoff_dataset_id = _literature_handoff_dataset_id(state)
    data_analysis_delegate = context.sub_agent_registry.resolve_by_delegate_function(
        "delegate_to_data_analysis"
    )
    if (
        handoff_dataset_id is not None
        and data_analysis_delegate is not None
        and state.get("sub_agent_call_count", 0)
        < context.settings.agent_max_tool_calls_per_turn
    ):
        model_calls = state.get("model_call_count", 0)
        call_id = context.id_generator.new_id()
        task = (
            f"对文献证据数据集 {handoff_dataset_id} 做描述统计。"
            "columns=numeric_value group_by=measurement_context。"
            "按论文、具体样品及指标单位隔离；未知条件不得视为相同条件。"
            "不同指标和单位不得合并；仅输出观察性结果，不推断因果。"
        )
        arguments_json = json.dumps({"task": task}, ensure_ascii=False)
        items = list(state.get("input_items", []))
        items.append(
            AgentFunctionCallItem(
                call_id=call_id,
                name=data_analysis_delegate.delegate_function_name,
                arguments=arguments_json,
            ).model_dump(mode="json")
        )
        return {
            "model_call_count": model_calls + 1,
            "current_node": node,
            "input_items": items,
            "pending_tool_calls": [
                {
                    "call_id": call_id,
                    "name": data_analysis_delegate.delegate_function_name,
                    "arguments_json": arguments_json,
                }
            ],
            "final_draft": None,
            "status": "running",
            "events": [
                _progress(
                    state,
                    context,
                    node=node,
                    event_type="literature_dataset_delegated",
                    message="文献数值证据已交给数据分析 Agent",
                    metrics={"dataset_id": handoff_dataset_id},
                )
            ],
        }
    passthrough = _materials_database_passthrough(state)
    if passthrough is not None:
        message_text = json.dumps(passthrough, ensure_ascii=False)
        items = list(state.get("input_items", []))
        items.append(
            AgentMessageItem(role="assistant", content=message_text).model_dump(
                mode="json"
            )
        )
        return {
            "current_node": node,
            "input_items": items,
            "pending_tool_calls": [],
            "final_draft": passthrough,
            "status": "running",
            "events": [
                _progress(
                    state,
                    context,
                    node=node,
                    event_type="materials_database_result_preserved",
                    message="Master 已保留材料数据库的完整查询结果",
                    metrics={"model_call": state.get("model_call_count", 0)},
                )
            ],
        }
    model_calls = state.get("model_call_count", 0)
    max_calls = context.settings.agent_max_model_calls_per_turn
    if model_calls >= max_calls:
        return _fail(
            state,
            context,
            node,
            "MODEL_CALL_LIMIT",
            f"model call limit {max_calls} reached",
        )
    input_bytes = _input_items_bytes(state.get("input_items", []))
    max_input_bytes = context.settings.agent_max_input_bytes
    if input_bytes > max_input_bytes:
        return _fail(
            state,
            context,
            node,
            "INPUT_TOO_LARGE",
            f"transcript exceeds {max_input_bytes} bytes",
        )
    # Build tool definitions from sub-agent registry
    sa_defs = context.sub_agent_registry.definitions_for_model()
    tool_definitions = tuple(
        AgentToolDefinition(
            name=d["name"],
            description=d["description"],
            parameters=d["parameters"],
            side_effect=ToolSideEffect.READ_ONLY,
            version="1",
        )
        for d in sa_defs
    )
    user_message = state.get("user_message", "").lower()
    data_analysis_delegate = context.sub_agent_registry.resolve_by_delegate_function(
        "delegate_to_data_analysis"
    )
    if (
        model_calls == 0
        and data_analysis_delegate is not None
        and _is_data_analysis_request(user_message)
    ):
        task = state.get("user_message", "")
        call_id = context.id_generator.new_id()
        items = list(state.get("input_items", []))
        arguments_json = json.dumps({"task": task}, ensure_ascii=False)
        items.append(
            AgentFunctionCallItem(
                call_id=call_id,
                name=data_analysis_delegate.delegate_function_name,
                arguments=arguments_json,
            ).model_dump(mode="json")
        )
        return {
            "model_call_count": model_calls + 1,
            "current_node": node,
            "input_items": items,
            "pending_tool_calls": [
                {
                    "call_id": call_id,
                    "name": data_analysis_delegate.delegate_function_name,
                    "arguments_json": arguments_json,
                }
            ],
            "final_draft": None,
            "status": "running",
            "events": [
                _progress(
                    state,
                    context,
                    node=node,
                    event_type="data_analysis_request_delegated",
                    message="data analysis request delegated",
                    metrics={},
                )
            ],
        }
    literature_delegate = context.sub_agent_registry.resolve_by_delegate_function(
        "delegate_to_literature"
    )
    if (
        model_calls == 0
        and literature_delegate is not None
        and _is_literature_evidence_request(user_message)
    ):
        task = state.get("user_message", "")
        call_id = context.id_generator.new_id()
        arguments_json = json.dumps({"task": task}, ensure_ascii=False)
        items = list(state.get("input_items", []))
        items.append(
            AgentFunctionCallItem(
                call_id=call_id,
                name=literature_delegate.delegate_function_name,
                arguments=arguments_json,
            ).model_dump(mode="json")
        )
        return {
            "model_call_count": model_calls + 1,
            "current_node": node,
            "input_items": items,
            "pending_tool_calls": [
                {
                    "call_id": call_id,
                    "name": literature_delegate.delegate_function_name,
                    "arguments_json": arguments_json,
                }
            ],
            "final_draft": None,
            "status": "running",
            "events": [
                _progress(
                    state,
                    context,
                    node=node,
                    event_type="literature_evidence_request_delegated",
                    message="合成科学问题已交给文献 Agent",
                    metrics={},
                )
            ],
        }
    database_delegate = context.sub_agent_registry.resolve_by_delegate_function(
        "delegate_to_materials_database"
    )
    if (
        model_calls == 0
        and database_delegate is not None
        and _is_material_database_request(user_message)
    ):
        task = state.get("user_message", "")
        previous_database_results = [
            result
            for result in state.get("sub_agent_results", [])
            if result.get("sub_agent_name") == "materials_database"
            and result.get("status") == "ok"
        ]
        if previous_database_results and any(
            marker in task.lower()
            for marker in ("刚才", "上一轮", "previous", "above", "these results")
        ):
            previous_task = next(
                (
                    candidate.get("task")
                    for candidate in reversed(previous_database_results)
                    if isinstance(candidate.get("task"), str)
                    and not any(
                        marker in candidate["task"].lower()
                        for marker in ("刚才", "上一轮", "previous", "above")
                    )
                ),
                None,
            )
            if isinstance(previous_task, str) and previous_task.strip():
                task = (
                    f"{task}\n上一轮材料数据库任务：{previous_task}。"
                    "优先复用已有 query_id；若旧会话没有快照，则先重建该查询。"
                )
        active_thread = state.get("active_workflow_thread_id")
        if (
            _requires_material_screening_chain(task)
            and _is_uv_detector_application(task)
            and parse_formula_screening(task) is None
        ):
            task += (
                "\nBAND_GAP_EXPLORATION_POOL：另建独立二元氧化物带隙探索池，"
                "保留严格池原条件与快照，不将探索池混入严格筛选统计。"
            )
        if active_thread and not _has_explicit_material_source(task):
            task = f"{task}\n分析工作流 thread {active_thread} 的候选材料。"
        if _requires_material_screening_chain(task):
            task = with_screening_handoff_scope(task)
        call_id = context.id_generator.new_id()
        items = list(state.get("input_items", []))
        items.append(
            AgentFunctionCallItem(
                call_id=call_id,
                name=database_delegate.delegate_function_name,
                arguments=json.dumps({"task": task}, ensure_ascii=False),
            ).model_dump(mode="json")
        )
        return {
            # This deterministic routing branch replaces the first model
            # decision. Count it so the post-sub-agent turn is final-only and
            # cannot delegate repeatedly.
            "model_call_count": model_calls + 1,
            "current_node": node,
            "input_items": items,
            "pending_tool_calls": [
                {
                    "call_id": call_id,
                    "name": database_delegate.delegate_function_name,
                    "arguments_json": json.dumps({"task": task}, ensure_ascii=False),
                }
            ],
            "final_draft": None,
            "status": "running",
            "events": [
                _progress(
                    state,
                    context,
                    node=node,
                    event_type="materials_database_request_delegated",
                    message="materials database request delegated",
                    metrics={},
                )
            ],
        }
    request = MaterialAgentRequest(
        instructions=_MASTER_SYSTEM_PROMPT,
        input_items=_transcript_items(state.get("input_items", [])),
        tool_definitions=tool_definitions,
        final_draft_schema=AgentFinalDraft.model_json_schema(),
        max_output_tokens=8192,
        reasoning_effort="none",
        temperature=0.0,
        allow_tool_calls=(
            state.get("sub_agent_call_count", 0)
            < context.settings.agent_max_tool_calls_per_turn
        ),
    )
    try:
        response = context.master_model.generate(request)
    except AgentModelError as exc:
        return _fail(state, context, node, "MODEL_ERROR", str(exc))

    model_attempt = model_calls + 1
    base: dict[str, Any] = {
        "model_call_count": model_attempt,
        "current_node": node,
    }
    if response.status is not AgentModelStatus.COMPLETED:
        code = (
            "MODEL_INCOMPLETE"
            if response.status is AgentModelStatus.INCOMPLETE
            else "MODEL_FAILED"
        )
        return _fail(
            state,
            context,
            node,
            code,
            response.error or "master model did not complete",
            base=base,
        )

    items = list(state.get("input_items", []))
    if response.tool_calls:
        tool_names = [call.name for call in response.tool_calls]
        for call in response.tool_calls:
            items.append(
                AgentFunctionCallItem(
                    call_id=call.call_id,
                    name=call.name,
                    arguments=call.arguments,
                ).model_dump(mode="json")
            )
        return {
            **base,
            "input_items": items,
            "pending_tool_calls": [
                {
                    "call_id": call.call_id,
                    "name": call.name,
                    "arguments_json": call.arguments,
                }
                for call in response.tool_calls
            ],
            "final_draft": None,
            "status": "running",
            "events": [
                _progress(
                    state,
                    context,
                    node=node,
                    event_type="model_responded",
                    message=(
                        f"Master 决定委派子Agent: {', '.join(tool_names)} "
                        f"（第 {model_attempt} 次模型调用）"
                    ),
                    metrics={"model_call": model_attempt},
                )
            ],
        }

    response_message_text = response.message_text
    if not response_message_text:
        return _fail(
            state,
            context,
            node,
            "MODEL_EMPTY_OUTPUT",
            "master model returned no final message",
            base=base,
        )
    try:
        draft = json.loads(response_message_text)
        AgentFinalDraft.model_validate(draft)
    except (json.JSONDecodeError, ValidationError):
        return _fail(
            state,
            context,
            node,
            "INVALID_FINAL_DRAFT",
            "master model final message is not a valid AgentFinalDraft",
            base=base,
        )
    items.append(
        AgentMessageItem(role="assistant", content=response_message_text).model_dump(
            mode="json"
        )
    )
    return {
        **base,
        "input_items": items,
        "pending_tool_calls": [],
        "final_draft": draft,
        "status": "running",
        "events": [
            _progress(
                state,
                context,
                node=node,
                event_type="model_final",
                message=(f"Master 已生成最终汇总回答（第 {model_attempt} 次模型调用）"),
                metrics={"model_call": model_attempt},
            )
        ],
    }


# ── Node: execute_sub_agent ───────────────────────────────────────────────────


def execute_sub_agent_node(
    state: MasterAgentState,
    runtime: Runtime[MasterAgentContext],
) -> dict[str, Any]:
    context = runtime.context
    node = NODE_EXECUTE_SUB_AGENT
    if state.get("error") is not None:
        return {}
    if _is_cancelled(context):
        return _cancel(state, context, node)
    pending = state.get("pending_tool_calls", [])
    if not pending:
        raise AgentInvariantError("execute_sub_agent requires pending_tool_calls")
    try:
        calls = [SubAgentCall.model_validate(raw) for raw in pending]
    except ValidationError as exc:
        raise AgentInvariantError("pending_tool_calls are invalid") from exc
    attempted_calls = state.get("sub_agent_call_count", 0) + len(calls)
    max_calls = context.settings.agent_max_tool_calls_per_turn
    if attempted_calls > max_calls:
        return _fail(
            state,
            context,
            node,
            "SUB_AGENT_CALL_LIMIT",
            f"sub-agent call limit {max_calls} reached",
        )

    outcomes = context.sub_agent_executor.execute(
        calls,
        parent_conversation_id=state.get("conversation_id", ""),
    )
    items = list(state.get("input_items", []))

    sub_names: list[str] = []
    evidence_ids: list[str] = list(state.get("evidence_ids", []))
    executed: list[str] = list(state.get("executed_call_ids", []))
    # ``sub_agent_results`` has an additive state reducer, so return only this
    # node's new results. Returning historical values here would duplicate them
    # on every follow-up turn.
    accumulated: list[dict[str, Any]] = []
    active_thread = state.get("active_workflow_thread_id")

    for outcome in outcomes:
        items.append(
            AgentFunctionOutputItem(
                call_id=outcome.call_id,
                output=outcome.output_json,
            ).model_dump(mode="json")
        )
        executed.append(outcome.call_id)
        sub_names.append(outcome.sub_agent_name)
        evidence_ids.append(outcome.call_id)
        accumulated.append(outcome.envelope.model_dump(mode="json"))
        returned_thread = outcome.envelope.active_workflow_thread_id
        if outcome.status == "ok" and returned_thread:
            conversation_id = state.get("conversation_id", "")
            context.conversation_store.link_workflow(conversation_id, returned_thread)
            context.conversation_store.set_active_thread(
                conversation_id, returned_thread
            )
            active_thread = returned_thread

    sub_agent_calls = state.get("sub_agent_call_count", 0) + len(calls)

    return {
        "input_items": items,
        "pending_tool_calls": [],
        "executed_call_ids": executed,
        "evidence_ids": evidence_ids,
        "sub_agent_call_count": sub_agent_calls,
        "sub_agent_results": accumulated,
        "active_workflow_thread_id": active_thread,
        "current_node": node,
        "status": "running",
        "events": [
            _progress(
                state,
                context,
                node=node,
                event_type="sub_agents_executed",
                message=(f"委派子Agent执行完成: {', '.join(sub_names)}"),
                metrics={"sub_agent_count": len(calls)},
            )
        ],
    }


# ── Node: validate_final ──────────────────────────────────────────────────────


def validate_final_node(
    state: MasterAgentState,
    runtime: Runtime[MasterAgentContext],
) -> dict[str, Any]:
    context = runtime.context
    node = NODE_VALIDATE_FINAL
    if state.get("error") is not None:
        return {}
    draft = state.get("final_draft")
    if draft is None:
        raise AgentInvariantError("validate_final requires final_draft")
    # Simple validation: draft must have status and answer keys
    if not isinstance(draft, dict):
        return _fail(
            state,
            context,
            node,
            "FINAL_VALIDATION_FAILED",
            "final draft is not a valid object",
        )
    status = draft.get("status")
    if status not in ("completed", "needs_user_input", "error"):
        return _fail(
            state,
            context,
            node,
            "FINAL_VALIDATION_FAILED",
            f"invalid final draft status: {status!r}",
        )
    if not isinstance(draft.get("answer"), str) or not draft["answer"].strip():
        return _fail(
            state,
            context,
            node,
            "FINAL_VALIDATION_FAILED",
            "final draft answer must not be empty",
        )
    return {
        "current_node": node,
        "events": [
            _progress(
                state,
                context,
                node=node,
                event_type="validation_passed",
                message="最终汇总回答验证通过",
                metrics={},
            )
        ],
    }


# ── Node: finalize_success ────────────────────────────────────────────────────


def finalize_success_node(
    state: MasterAgentState,
    runtime: Runtime[MasterAgentContext],
) -> dict[str, Any]:
    context = runtime.context
    node = NODE_FINALIZE_SUCCESS
    draft = state.get("final_draft")
    if draft is None:
        raise AgentInvariantError("finalize_success requires final_draft")
    answer = draft.get("answer", "")
    return {
        "status": "completed",
        "current_node": node,
        "final_response": answer,
        "pending_tool_calls": [],
        "error": None,
        "events": [
            _event(
                state,
                context,
                node=node,
                event_type="turn_completed",
                status="completed",
                message="master agent turn completed",
                metrics={
                    "model_call_count": state.get("model_call_count", 0),
                    "sub_agent_call_count": state.get("sub_agent_call_count", 0),
                },
            )
        ],
    }


# ── Node: finalize_error ──────────────────────────────────────────────────────


def finalize_error_node(
    state: MasterAgentState,
    runtime: Runtime[MasterAgentContext],
) -> dict[str, Any]:
    context = runtime.context
    node = NODE_FINALIZE_ERROR
    error = state.get("error") or {
        "code": "UNKNOWN_ERROR",
        "message": "unknown master agent error",
        "retryable": False,
    }
    return {
        "status": "error",
        "current_node": node,
        "final_response": str(error.get("message", "unknown master agent error")),
        "final_draft": None,
        "pending_tool_calls": [],
        "error": error,
        "events": [
            _event(
                state,
                context,
                node=node,
                event_type="turn_error",
                status="error",
                message="master agent turn failed",
                metrics={
                    "error_code": str(error.get("code", "UNKNOWN_ERROR")),
                },
            )
        ],
    }


# ── Node: finalize_cancelled ──────────────────────────────────────────────────


def finalize_cancelled_node(
    state: MasterAgentState,
    runtime: Runtime[MasterAgentContext],
) -> dict[str, Any]:
    context = runtime.context
    node = NODE_FINALIZE_CANCELLED
    return {
        "status": "cancelled",
        "current_node": node,
        "final_response": "已停止：不再执行后续步骤（进行中的步骤将完成）",
        "final_draft": None,
        "pending_tool_calls": [],
        "error": None,
        "events": [
            _event(
                state,
                context,
                node=node,
                event_type="turn_cancelled",
                status="cancelled",
                message="master agent turn cancelled by user",
                metrics={
                    "model_call_count": state.get("model_call_count", 0),
                    "sub_agent_call_count": state.get("sub_agent_call_count", 0),
                },
            )
        ],
    }


# ── Routing ────────────────────────────────────────────────────────────────────


def route_after_model(state: MasterAgentState) -> RouteAfterModel:
    if state.get("error") is not None:
        return "finalize_error"
    if state.get("cancelled"):
        return "finalize_cancelled"
    if state.get("pending_tool_calls"):
        return "execute_sub_agent"
    if state.get("final_draft") is not None:
        return "validate_final"
    raise AgentInvariantError(
        "route_after_model requires error, pending_tool_calls or final_draft"
    )


def route_after_sub_agent(state: MasterAgentState) -> RouteAfterSubAgent:
    if state.get("error") is not None:
        return "finalize_error"
    if state.get("cancelled"):
        return "finalize_cancelled"
    return "call_master_model"


def route_after_validation(state: MasterAgentState) -> RouteAfterValidation:
    if state.get("error") is not None:
        return "finalize_error"
    return "finalize_success"


# ── Helpers ────────────────────────────────────────────────────────────────────


def _transcript_items(
    items: list[dict[str, Any]],
) -> tuple[AgentMessageItem | AgentFunctionCallItem | AgentFunctionOutputItem, ...]:
    converted: list[
        AgentMessageItem | AgentFunctionCallItem | AgentFunctionOutputItem
    ] = []
    for raw in items:
        item_type = raw.get("type")
        if item_type == "message":
            converted.append(AgentMessageItem.model_validate(raw))
        elif item_type == "function_call":
            converted.append(AgentFunctionCallItem.model_validate(raw))
        elif item_type == "function_call_output":
            converted.append(AgentFunctionOutputItem.model_validate(raw))
        else:
            raise AgentInvariantError(
                f"unsupported transcript item type: {item_type!r}"
            )
    return tuple(converted)


def _current_turn_sub_agent_results(
    state: MasterAgentState,
) -> list[dict[str, Any]]:
    current_call_ids = set(state.get("executed_call_ids", []))
    return [
        result
        for result in state.get("sub_agent_results", [])
        if result.get("call_id") in current_call_ids
    ]


def _queue_sub_agent_delegation(
    state: MasterAgentState,
    context: MasterAgentContext,
    *,
    node: str,
    delegate_name: str,
    task: str,
    event_type: str,
    message: str,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """Queue one deterministic cross-agent handoff."""

    call_id = context.id_generator.new_id()
    arguments_json = json.dumps({"task": task}, ensure_ascii=False)
    items = list(state.get("input_items", []))
    items.append(
        AgentFunctionCallItem(
            call_id=call_id,
            name=delegate_name,
            arguments=arguments_json,
        ).model_dump(mode="json")
    )
    return {
        "model_call_count": state.get("model_call_count", 0) + 1,
        "current_node": node,
        "input_items": items,
        "pending_tool_calls": [
            {
                "call_id": call_id,
                "name": delegate_name,
                "arguments_json": arguments_json,
            }
        ],
        "final_draft": None,
        "status": "running",
        "events": [
            _progress(
                state,
                context,
                node=node,
                event_type=event_type,
                message=message,
                metrics=metrics,
            )
        ],
    }


def _material_query_handoff_id(state: MasterAgentState) -> str | None:
    """Return the current turn's reproducible database query snapshot."""

    for result in reversed(_current_turn_sub_agent_results(state)):
        if (
            result.get("sub_agent_name") != "materials_database"
            or result.get("status") != "ok"
        ):
            continue
        response_text = result.get("response_text")
        if not isinstance(response_text, str):
            continue
        match = _MATERIAL_QUERY_HANDOFF_PATTERN.search(response_text)
        if match is not None:
            return match.group(1)
    return None


def _supplementary_query_handoff_id(state: MasterAgentState) -> str | None:
    for result in reversed(_current_turn_sub_agent_results(state)):
        if (
            result.get("sub_agent_name") == "materials_database"
            and result.get("status") == "ok"
        ):
            match = re.search(
                r"SUPPLEMENTARY_QUERY_HANDOFF:\s*query_id=(query-[A-Za-z0-9-]+)",
                str(result.get("response_text", "")),
            )
            if match:
                return match.group(1)
    return None


def _material_screening_partial_report(
    state: MasterAgentState,
    *,
    call_limit: int,
) -> dict[str, Any]:
    """Preserve actual outputs while making incomplete stages explicit.

    Failure call IDs are audit references, never successful evidence. This
    renderer does not recover or infer a failed sub-agent's internal results.
    """

    results = _current_turn_sub_agent_results(state)
    used = state.get("sub_agent_call_count", 0)
    reason = (
        f"本轮子Agent调用额度已用完（{used}/{call_limit}）；未执行步骤不再委派。"
        if used >= call_limit
        else f"后续交接条件不足，调用计数为 {used}/{call_limit}；"
        "失败步骤本轮不自动重试，未执行步骤不能视为完成。"
    )
    sections = [
        f"原始科研问题：{state.get('user_message', '')}",
        "这是部分执行报告，不是完整链路成功报告。保留本轮真实返回，"
        "不以质量检查代替描述统计，也不以失败响应补造结论。",
        reason,
    ]
    query_id = _material_query_handoff_id(state)
    if query_id:
        sections.append(f"已保存的数据库查询快照：`{query_id}`。")
    evidence_ids: list[str] = []
    material_ids: list[str] = []
    warnings: list[str] = []
    for name, label, limit in (
        ("materials_database", "数据库初筛", 1600),
        ("data_analysis", "数据分析", 1200),
        ("literature", "文献检索", 3500),
    ):
        attempts = [item for item in results if item.get("sub_agent_name") == name]
        successful = next(
            (
                item
                for item in reversed(attempts)
                if item.get("status") == "ok"
                and isinstance(item.get("response_text"), str)
                and item["response_text"].strip()
            ),
            None,
        )
        for attempt in attempts:
            warnings.extend(
                warning
                for warning in attempt.get("warnings", [])
                if isinstance(warning, str) and warning.strip()
            )
            error = attempt.get("error")
            if attempt.get("status") == "error" and isinstance(error, dict):
                detail = error.get("message")
                if isinstance(detail, str) and detail.strip():
                    warnings.append(f"{label}：{detail}")
        if successful is not None:
            text = _MATERIAL_QUERY_HANDOFF_PATTERN.sub(
                "", successful["response_text"]
            ).strip()
            sections.append(
                f"## {label}：已返回结果\n\n{_bounded_section(text, limit)}"
            )
            call_id = successful.get("call_id")
            if isinstance(call_id, str) and call_id:
                evidence_ids.append(call_id)
            if name == "materials_database":
                material_ids.extend(re.findall(r"\bmp-[A-Za-z0-9-]+\b", text))
        elif attempts:
            failed = attempts[-1]
            error = failed.get("error")
            detail = (
                str(error.get("message") or error.get("code") or "原因未返回")
                if isinstance(error, dict)
                else "没有返回可用结果"
            )
            sections.append(
                f"## {label}：执行失败\n\n{_bounded_section(detail, 500)}\n\n"
                f"调用记录：`{failed.get('call_id', '')}`（不是成功证据）。"
            )
        else:
            sections.append(f"## {label}：未执行\n\n没有本轮返回结果，不等于零条命中。")
    sections.append(
        "## 结论边界与下一步\n\n"
        "成功结果与查询快照仍保留。需先处理失败原因或恢复交接条件，"
        "再明确重试未完成步骤；本轮不会自动提高调用上限或重复失败步骤。"
        "计算属性仅支持初筛，文献题名/摘要仅是线索。"
        "目前不能确认最终前几名、全文实验结论或目标应用可行性。"
    )
    warnings.append("链路未完整完成，部分结果不能作为最终材料推荐。")
    return AgentFinalDraft(
        status=AgentFinalStatus.ERROR,
        answer="\n\n".join(sections),
        referenced_material_ids=list(dict.fromkeys(material_ids)),
        evidence_ids=list(dict.fromkeys(evidence_ids)),
        warnings=list(dict.fromkeys(warnings)),
        follow_up_question=None,
    ).model_dump(mode="json")


def _material_screening_chain_passthrough(
    state: MasterAgentState,
    *,
    coordinator: DataAnalysisCrossAgentCoordinator | None = None,
) -> dict[str, Any] | None:
    """Combine database, analysis and literature outputs without inventing rank."""

    if not _requires_material_screening_chain(state.get("user_message", "")):
        return None
    results = _current_turn_sub_agent_results(state)
    selected: dict[str, dict[str, Any]] = {}
    for name in ("materials_database", "data_analysis"):
        result = next(
            (
                item
                for item in results
                if item.get("sub_agent_name") == name
                and item.get("status") == "ok"
                and isinstance(item.get("response_text"), str)
                and item["response_text"].strip()
            ),
            None,
        )
        if result is None:
            return None
        selected[name] = result
    literature = next(
        (
            item
            for item in results
            if item.get("sub_agent_name") == "literature"
            and item.get("status") in {"ok", "error"}
            and (
                item.get("status") == "error"
                or (
                    isinstance(item.get("response_text"), str)
                    and item["response_text"].strip()
                )
            )
        ),
        None,
    )
    if literature is None:
        return None
    selected["literature"] = literature

    database_text = _MATERIAL_QUERY_HANDOFF_PATTERN.sub(
        "", str(selected["materials_database"]["response_text"])
    ).strip()
    database_text = re.sub(
        r"SUPPLEMENTARY_QUERY_HANDOFF:\s*query_id=query-[A-Za-z0-9-]+",
        "",
        database_text,
    ).strip()
    database_text = re.sub(
        r"匹配\s*2000\s*条",
        "本次检索达到 2000 条返回上限（不代表数据库总命中数）",
        database_text,
    )
    analysis_text = str(selected["data_analysis"]["response_text"]).strip()
    literature_failed = selected["literature"].get("status") == "error"
    literature_error = selected["literature"].get("error")
    literature_error_detail = (
        literature_error.get("message") if isinstance(literature_error, dict) else None
    )
    if literature_failed:
        application_label = (
            "紫外探测"
            if _is_uv_detector_application(state.get("user_message", ""))
            else "目标应用"
        )
        literature_text = (
            "文献核验本轮执行失败：具体原因见警告。数据库初筛和"
            f"描述统计仍然保留，但本轮不能形成{application_label}可行性结论。可在文献源"
            "恢复后用同一问题重试，无需重新修改筛选条件。"
        )
    else:
        literature_text = str(selected["literature"]["response_text"]).strip()
    queue_text = "应用优先队列不可用；保留原始数据库结果供人工核验。"
    query_id = _material_query_handoff_id(state)
    supplementary_id = _supplementary_query_handoff_id(state)
    supplementary_note = (
        f"\n独立探索池快照：`{supplementary_id}`；严格池快照：`{query_id}`。"
        "探索池查 0 < 计算带隙 ≤ 原上限的二元氧化物，保留凸包能与禁用元素条件。"
        "它不是原严格条件达标名单，不混入第 3 节统计，也不采用固定带隙校正。\n"
        if supplementary_id
        else ""
    )
    uv_application = _is_uv_detector_application(state.get("user_message", ""))
    if uv_application and coordinator is not None and query_id is not None:
        try:
            candidates = coordinator.prioritized_material_query_candidates(
                query_id, limit=5
            )
        except ValueError:
            candidates = ()
        if candidates:
            queue_text = render_uv_candidate_queue(candidates)
    answer = (
        "## 1. Materials Project 初筛\n\n"
        f"{_bounded_section(database_text, 1600)}\n\n"
        "## 2. 数据库属性初排（文献预检池前 5 名）\n\n"
        f"{queue_text}\n\n"
        "该队列不修改原始数据库快照。放射性锕系仅在应用队列中硬排除；"
        "As/Tl/Be 与稀土只作风险提示，不被静默删除。`theoretical=false` "
        "不等于已有紫外探测器实验。\n\n"
        "## 3. 候选数据分析\n\n"
        f"{_bounded_section(analysis_text, 1200)}\n\n"
        "## 4. 候选池文献预检、证据重排与下载清单\n\n"
        f"{supplementary_note}\n{_bounded_section(literature_text, 3500)}\n\n"
        "## 5. 结论边界\n\n"
        "第 2 节的前 5 名只是数据库属性初排示例，不是最终推荐。最终重点候选"
        "必须经过第 4 节的扩大候选池文献预检与证据重排。Materials Project 的"
        "带隙、凸包能和密度只能支持初筛；"
        "当前候选查询最多接收 2000 条返回、分析快照最多保存前 1000 条，因此"
        "统计与重点候选都不是对 Materials Project 全库的穷尽结论。最终可行性必须"
        "由材料对应关系明确的实验器件文献或后续实验确认。A 级文献证据必须"
        "同时出现实际紫外探测器与器件指标；B/C 级只能作为"
        "光学实验背景或一般材料背景，不能单独支持‘适合紫外探测’的结论。"
        "当前仍是题名/摘要预检，下一阶段需下载并上传相应 PDF，核对物相、实验条件"
        "和指标后才能形成全文证据报告。"
    )
    if not uv_application:
        preview = "候选快照预览不可用；保留第1节数据库结果。"
        if coordinator is not None and query_id is not None:
            try:
                raw_candidates = coordinator.material_query_candidates(
                    query_id, limit=5
                )
            except ValueError:
                raw_candidates = ()
            if raw_candidates:
                preview = "\n".join(
                    ["| Material ID | 化学式 |", "|---|---|"]
                    + [
                        f"| {row['material_id']} | {row['formula_pretty']} |"
                        for row in raw_candidates
                    ]
                )
        answer = (
            f"原始科研问题：{state.get('user_message', '')}\n\n"
            "## 1. Materials Project 初筛\n\n"
            f"{_bounded_section(database_text, 1600)}\n\n"
            "## 2. 数据库候选快照预览\n\n"
            f"{preview}\n\n"
            "预览仅按查询快照顺序展示前5条，不是应用性能排名，也不是最终推荐。"
            "通用文献检索从完整快照按化学式去重，再按各化学式最低凸包能"
            "安排最多20种的检索队列；这不是性能排名，也不代表全候选池核验。\n\n"
            "## 3. 候选数据分析\n\n"
            f"{_bounded_section(analysis_text, 1200)}\n\n"
            "## 4. 目标应用文献检索与证据缺口\n\n"
            f"{_bounded_section(literature_text, 3500)}\n\n"
            "## 5. 结论边界与下一步\n\n"
            "Materials Project 的计算带隙、凸包能与密度只支持初筛，"
            "不能替代目标应用的实验指标。候选查询最多接收2000条返回，"
            "分析快照最多保存前1000条，不是对全库的穷尽结论。"
            "化学式相同也不代表论文样品与 MP 物相相同；theoretical=false "
            "不等于目标应用已验证。文献题名/摘要不是全文实验依据，"
            "尚不能确认最终前几名或材料应用可行性。"
            "需取得对应论文 PDF 正文，核对物相、实验条件、指标与材料角色后继续。"
        )
    warnings = list(
        dict.fromkeys(
            [
                warning
                for result in selected.values()
                for warning in result.get("warnings", [])
                if isinstance(warning, str) and warning.strip()
            ]
            + (
                [literature_error_detail]
                if isinstance(literature_error_detail, str)
                and literature_error_detail.strip()
                else []
            )
            + ["文献检索结果属于证据核验层，不等同于实验复现或专家确认。"]
        )
    )
    material_ids = list(
        dict.fromkeys(re.findall(r"\bmp-[A-Za-z0-9-]+\b", database_text))
    )
    call_ids = [
        str(selected[name]["call_id"])
        for name in ("materials_database", "data_analysis", "literature")
        if isinstance(selected[name].get("call_id"), str) and selected[name]["call_id"]
    ]
    needs_pdf = not uv_application and not literature_failed
    follow_up = (
        "请提供第4节中相关论文的 PDF 正文，以继续全文证据核验；"
        "若本轮没有找到相关论文，需要补充检索或提供已有论文，不能直接确认推荐。"
        if needs_pdf
        else "请下载清单中可获得的论文并上传 PDF，以继续全文证据核验。"
        if "请下载上述" in literature_text
        else None
    )
    draft = AgentFinalDraft(
        # General-topic discovery is an intermediate metadata report, not a
        # completed full-text feasibility workflow.
        status=AgentFinalStatus.ERROR
        if literature_failed
        else AgentFinalStatus.NEEDS_USER_INPUT
        if follow_up is not None
        else AgentFinalStatus.COMPLETED,
        answer=answer,
        referenced_material_ids=material_ids,
        evidence_ids=call_ids,
        warnings=warnings,
        follow_up_question=follow_up,
    )
    return draft.model_dump(mode="json")


def _bounded_section(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return (
        text[:limit].rsplit("\n", 1)[0]
        + "\n\n（篇幅限制作摘要展示；完整结果保留在对应 Agent 会话与快照。）"
    )


def _literature_handoff_dataset_id(state: MasterAgentState) -> str | None:
    """Return a dataset staged by this turn's successful LiteratureAgent."""

    for result in reversed(_current_turn_sub_agent_results(state)):
        if result.get("sub_agent_name") != "literature" or result.get("status") != "ok":
            continue
        response_text = result.get("response_text")
        if not isinstance(response_text, str):
            continue
        match = _LITERATURE_HANDOFF_PATTERN.search(response_text)
        if match is not None:
            return match.group(1)
    return None


def _literature_analysis_passthrough(
    state: MasterAgentState,
) -> dict[str, Any] | None:
    """Combine the evidence report and its downstream analysis deterministically."""

    results = _current_turn_sub_agent_results(state)
    literature = next(
        (
            result
            for result in results
            if result.get("sub_agent_name") == "literature"
            and result.get("status") == "ok"
            and isinstance(result.get("response_text"), str)
            and _LITERATURE_HANDOFF_PATTERN.search(result["response_text"])
        ),
        None,
    )
    analysis = next(
        (
            result
            for result in results
            if result.get("sub_agent_name") == "data_analysis"
            and result.get("status") == "ok"
            and isinstance(result.get("response_text"), str)
        ),
        None,
    )
    if literature is None or analysis is None:
        return None

    literature_text = _LITERATURE_HANDOFF_PATTERN.sub(
        "数值证据已交给数据分析 Agent",
        str(literature["response_text"]),
    )
    answer = (
        "## 文献 Agent 结果\n\n"
        f"{_bounded_section(literature_text, 4800)}\n\n"
        "## 数据分析 Agent 结果\n\n"
        f"{_bounded_section(str(analysis['response_text']), 1200)}\n\n"
        "## 结论边界\n\n"
        "本轮数据来自公开文献证据，只能用于验证系统链路和"
        "观察性综合；不能替代独立实验数据，也不能据此推断因果。"
    )
    warnings = list(
        dict.fromkeys(
            warning
            for result in (literature, analysis)
            for warning in result.get("warnings", [])
            if isinstance(warning, str) and warning.strip()
        )
    )
    if warnings:
        risk_text = "\n".join(f"- {warning}" for warning in warnings)
        answer += "\n\n## 风险提示\n\n" + _bounded_section(risk_text, 1200)
    call_ids = [
        str(result["call_id"])
        for result in (literature, analysis)
        if isinstance(result.get("call_id"), str) and result["call_id"]
    ]
    draft = AgentFinalDraft(
        status=AgentFinalStatus.COMPLETED,
        answer=answer,
        referenced_material_ids=[],
        evidence_ids=call_ids,
        warnings=warnings,
        follow_up_question=None,
    )
    return draft.model_dump(mode="json")


def _materials_database_passthrough(
    state: MasterAgentState,
) -> dict[str, Any] | None:
    """Preserve one successful database or literature answer verbatim.

    A second generative pass can silently discard tables, rows, caveats, and
    units.  The outer delegation call remains the Master's evidence boundary,
    so the normal final validator still checks this deterministic draft.
    """
    # ``sub_agent_results`` is intentionally accumulated across conversation
    # turns.  Only results whose delegation call ran in this user turn may be
    # published; otherwise a follow-up (statistics/outliers/export) can replay
    # the previous search answer before it is even delegated.
    if state.get("model_call_count", 0) == 0:
        return None
    results = _current_turn_sub_agent_results(state)
    if len(results) != 1:
        return None
    result = results[0]
    sub_agent_name = result.get("sub_agent_name")
    if sub_agent_name not in {
        "materials_database",
        "literature",
    }:
        return None
    if result.get("status") != "ok":
        if sub_agent_name != "literature":
            return None
        error = result.get("error")
        detail = error.get("message") if isinstance(error, dict) else None
        answer = (
            "文献检索未能完成，但你的检索主题和时间范围已经足够明确，"
            "无需重新描述任务。请稍后直接重试。"
        )
        warnings = [str(detail)] if isinstance(detail, str) and detail else []
        draft = AgentFinalDraft(
            status=AgentFinalStatus.COMPLETED,
            answer=answer,
            referenced_material_ids=[],
            evidence_ids=[],
            warnings=warnings,
            follow_up_question=None,
        )
        return draft.model_dump(mode="json")
    response_text = result.get("response_text")
    if not isinstance(response_text, str) or not response_text.strip():
        return None

    if sub_agent_name == "materials_database":
        # The Materials Project retrieval layer currently caps broad result sets
        # at 2,000.  Do not let a model present that cap as the database total.
        response_text = re.sub(
            r"数据库中共有约\s*2000\s*个(?:符合条件的)?",
            "本次检索达到 2000 条返回上限（不代表数据库总数），其中",
            response_text,
        )
        response_text = _MATERIAL_QUERY_HANDOFF_PATTERN.sub(
            "查询快照已保存，可供后续数据分析复用。", response_text
        )
    else:
        response_text = _LITERATURE_HANDOFF_PATTERN.sub(
            "数值证据已准备，但本轮未执行数据分析",
            response_text,
        )
        if len(response_text) > 8000:
            risk_text = "\n".join(
                f"- {warning}"
                for warning in result.get("warnings", [])
                if isinstance(warning, str)
            )
            footer = "\n\n## 风险提示\n\n" + _bounded_section(risk_text, 1600)
            response_text = _bounded_section(response_text, 5900) + footer
    material_ids = (
        list(
            dict.fromkeys(
                match.group(0)
                for match in re.finditer(r"\bmp-[A-Za-z0-9-]+\b", response_text)
            )
        )
        if sub_agent_name == "materials_database"
        else []
    )
    warnings = [
        warning
        for warning in result.get("warnings", [])
        if isinstance(warning, str) and warning.strip()
    ]
    call_id = result.get("call_id")
    evidence_ids = [call_id] if isinstance(call_id, str) and call_id else []
    active_thread = result.get("active_workflow_thread_id")
    draft = AgentFinalDraft(
        status=AgentFinalStatus.COMPLETED,
        answer=response_text,
        active_workflow_thread_id=(
            active_thread if isinstance(active_thread, str) else None
        ),
        referenced_material_ids=material_ids,
        evidence_ids=evidence_ids,
        warnings=warnings,
        follow_up_question=None,
    )
    return draft.model_dump(mode="json")


def _input_items_bytes(items: list[dict[str, Any]]) -> int:
    try:
        return len(json.dumps(items, ensure_ascii=False).encode("utf-8"))
    except (TypeError, ValueError):
        return 0


def _is_cancelled(context: MasterAgentContext) -> bool:
    return context.cancel_event is not None and context.cancel_event.is_set()


def _cancel(
    state: MasterAgentState,
    context: MasterAgentContext,
    node: str,
) -> dict[str, Any]:
    return {
        "cancelled": True,
        "status": "running",
        "current_node": node,
        "pending_tool_calls": [],
        "final_draft": None,
        "events": [
            _progress(
                state,
                context,
                node=node,
                event_type="cancelled",
                message="检测到停止请求，正在结束本轮对话",
                metrics={},
            )
        ],
    }


def _fail(
    state: MasterAgentState,
    context: MasterAgentContext,
    node: str,
    code: str,
    message: str,
    *,
    base: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "error": {"code": code, "message": message, "retryable": False},
        "status": "error",
        "cancelled": False,
        "current_node": node,
        "final_response": message,
        "final_draft": None,
        "pending_tool_calls": [],
    }
    if base:
        payload.update(base)
    return payload


def _event(
    state: MasterAgentState,
    context: MasterAgentContext,
    *,
    node: str,
    event_type: str,
    status: str,
    message: str,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "event_id": context.id_generator.new_id(),
        "conversation_id": state.get("conversation_id", ""),
        "node": node,
        "event_type": event_type,
        "created_at": context.clock().isoformat(),
        "status": status,
        "message": message,
        "metrics": metrics,
    }


def _progress(
    state: MasterAgentState,
    context: MasterAgentContext,
    *,
    node: str,
    event_type: str,
    message: str,
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "event_id": f"{node}:{event_type}",
        "conversation_id": state.get("conversation_id", ""),
        "node": node,
        "event_type": event_type,
        "created_at": context.clock().isoformat(),
        "status": "running",
        "message": message,
        "metrics": metrics or {},
    }


def _parse_json_object(raw: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    return payload if isinstance(payload, dict) else None
