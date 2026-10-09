"""Gradio web UI — clean three-panel layout.

Launch: uv run materials-screen agent ui
"""

from __future__ import annotations

import json
import os
import random
import re
import sqlite3
import threading
from collections.abc import Generator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import gradio as gr
from dotenv import load_dotenv
from pydantic import SecretStr

load_dotenv()

# ── Globals ─────────────────────────────────────────────────────────────────
_global_runner: Any = None
_run_root = Path("data/workflow_runs")
_saved_results: list[dict[str, Any]] = []
# Per-run UI session state: the active conversation id is reused across
# messages so follow-up questions keep their multi-turn context; the cancel
# event enables cooperative stop; the in-progress flag rejects double submits.
_conversation_id: str | None = None
_cancel_event: threading.Event | None = None
_turn_in_progress = False

# ── Constants ───────────────────────────────────────────────────────────────
NODE_LABELS = {
    "prepare_turn": "准备对话",
    "call_agent_model": "调用 AI 模型",
    "execute_tools": "执行工具",
    "validate_final": "验证回答",
    "finalize_success": "完成",
    "finalize_error": "错误",
}

TOOL_LABELS = {
    "run_screening_workflow": "筛选工作流",
    "get_workflow_status": "获取状态",
    "get_workflow_history": "获取历史",
    "get_screening_result": "获取结果",
    "compare_ranked_materials": "比较材料",
}

QA_PROMPTS = [
    "寻找不含 Pb 的半导体材料",
    "寻找带隙在 1.0 到 2.0 eV 之间的稳定氧化物",
    "有哪些不含铅的钙钛矿材料？",
    "寻找含 Cu 且能量高于 hull 小于 0.1 eV/atom 的材料",
    "查找可用于光伏的 n 型透明导电材料",
    "寻找带隙大于 2 eV 的非金属氧化物",
    "有哪些立方晶系的半导体？",
    "比较 mp-1 和 mp-149 两种材料",
    "寻找不含稀土元素的磁性材料",
    "哪些材料适合做热电转换？",
    "寻找含 Zn 且不含 Cd 的半导体",
    "查找 band gap 在 1.5 到 2.5 之间的直接带隙材料",
    "筛选稳定的锂离子电池正极材料",
    "寻找适合做光催化水分解的材料",
    "寻找密度小于 5 g/cm³ 的轻质结构材料",
    "筛选可用于红外探测的窄带隙半导体",
    "有哪些含 S 且不含 Se 的硫化物？",
    "寻找适合做透明电极的材料",
    "查找热导率低的材料用于隔热",
    "筛选电子迁移率高的二维材料",
]

_NON_MATERIAL_MARKERS = (
    "会话",
    "chat",
    "撤回",
    "报错",
    "失败",
    "怎么办",
    "怎么调",
    "怎么开",
    "怎么用",
    "怎么让它",
    "怎么记住",
    "怎么补",
    "能读吗",
    "能删",
    "删掉",
    "能直接查",
    "上网",
    "web_search",
    "Python",
    "模式",
    "mock",
    "intern",
    "工具",
    "不调用",
    "DFT",
    "从头",
    "MODEL_ERROR",
    "connection",
    "FINAL_VALIDATION",
    "未安装",
    "安装",
    "重复",
    "还记得",
    "上一轮",
    "哪一步",
    "步骤",
    "状态",
    "任务",
    "验证",
    "密钥",
    "KEY",
    "读另一个",
    "太长",
    "最多",
    "新手",
    "开始用",
    "先试",
    "答非所问",
    "需要澄清",
    "一些材料",
    "合适的材料",
    "且小于",
    "前 10",
    "前10",
)

_QA_HEADING = re.compile(r"^### Q\d+\s+(.+?)\s*$")


def _load_qa_prompts() -> list[str]:
    """Load material-related quick prompts from the QA doc.

    Parses ``docs/qa/MATERIALS_QA.md`` and returns the question headings of
    the "Agent 真实场景问答" section (E) only, dropping usage /
    troubleshooting / security questions so only material questions remain.
    Returns an empty list when the document is missing so the caller can fall
    back to ``QA_PROMPTS``.
    """
    qa_path = (
        Path(__file__).resolve().parent.parent.parent
        / "docs"
        / "qa"
        / "MATERIALS_QA.md"
    )
    prompts: list[str] = []
    in_agent_section = False
    try:
        lines = qa_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        if line.startswith("## E."):
            in_agent_section = True
            continue
        if in_agent_section and line.startswith("## "):
            break
        if not in_agent_section:
            continue
        match = _QA_HEADING.match(line)
        if not match:
            continue
        prompt = match.group(1)
        if any(marker in prompt for marker in _NON_MATERIAL_MARKERS):
            continue
        prompts.append(prompt)
    return list(dict.fromkeys(prompts))


# ── Styling (dark theme: deep blue-black surfaces, teal accent) ─────────────
# Gradio paints body via "@media (prefers-color-scheme: dark) { body {...} }"
# and swaps variables under ":root.dark", so the theme is forced with
# !important declarations that win regardless of the OS color scheme.
CSS = """
footer { visibility: hidden; }

body {
    background: #0d1526 !important;
    color: #e2e8f0 !important;
}
:root:root {
    color-scheme: dark;
    --body-background-fill: #0d1526 !important;
    --background-fill-primary: #0d1526 !important;
    --background-fill-secondary: #111b33 !important;
    --block-background-fill: #16213a !important;
    --block-border-color: #2b3a55 !important;
    --block-shadow: 0 1px 3px rgba(0, 0, 0, 0.35) !important;
    --body-text-color: #e2e8f0 !important;
    --body-text-color-subdued: #94a3b8 !important;
    --block-label-text-color: #94a3b8 !important;
    --input-background-fill: #1e293b !important;
    --input-border-color: #334155 !important;
    --border-color-primary: #334155 !important;
    --primary-400: #5eead4 !important;
    --primary-500: #2dd4bf !important;
    --primary-600: #14b8a6 !important;
}

body, .gradio-container, .app, button, input, textarea, select,
.markdown, .prose, .chatbot {
    font-family: "Segoe UI", "Microsoft YaHei", "PingFang SC",
                 "Helvetica Neue", Arial, sans-serif !important;
}

/* ── Container: fixed three-column layout ────────────────────────────────── */
/* The left/middle/right panels always stay side by side; narrower windows
   scroll horizontally instead of re-stacking the columns. */
.gradio-container {
    max-width: 1400px !important;
    min-width: 1180px !important;
    margin: 0 auto !important;
    padding: 0.6rem 1rem 2rem !important;
}

/* ── Header banner ──────────────────────────────────────────────────────── */
#app-header {
    padding: 1.1rem 1.4rem;
    margin-bottom: 1rem;
    background: linear-gradient(120deg, rgba(45, 212, 191, 0.13),
                                rgba(59, 130, 246, 0.08));
    border: 1px solid #2b3a55;
    border-left: 5px solid #2dd4bf;
    border-radius: 12px;
}
#app-header h1 {
    font-size: 1.45rem;
    margin: 0 0 0.25rem;
    color: #f1f5f9;
}
#app-header p {
    margin: 0;
    color: #94a3b8;
    font-size: 0.9rem;
}

/* ── Blocks: deep cards ─────────────────────────────────────────────────── */
.gradio-container .block {
    border-radius: 12px !important;
}
.gradio-container .form {
    border-radius: 10px !important;
    border-color: #334155 !important;
    background: #1e293b !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.4) !important;
}
.gradio-container .block.gradio-accordion {
    border-color: #2b3a55 !important;
    border-radius: 10px !important;
}

/* ── Chat bubbles (Gradio 6 DOM: .message-row.bubble.user/bot-row) ───────── */
.bubble-wrap .message-row.bubble.user-row {
    justify-content: flex-end;
    margin-right: 8px;
    max-width: 85%;
}
.bubble-wrap .message-row.bubble.bot-row {
    margin-left: 8px;
    max-width: 85%;
}
.bubble-wrap .user.message {
    background: linear-gradient(135deg, #2dd4bf, #14b8a6) !important;
    color: #042f2e !important;
    border: none !important;
    border-radius: 14px 14px 4px 14px !important;
    box-shadow: 0 2px 8px rgba(45, 212, 191, 0.28) !important;
}
.bubble-wrap .bot.message {
    background: #1e293b !important;
    color: #e2e8f0 !important;
    border: 1px solid #334155 !important;
    border-radius: 14px 14px 14px 4px !important;
}

/* ── Buttons ────────────────────────────────────────────────────────────── */
.gradio-container button.primary {
    background: linear-gradient(135deg, #2dd4bf, #14b8a6) !important;
    border: none !important;
    color: #042f2e !important;
    box-shadow: 0 2px 8px rgba(45, 212, 191, 0.3) !important;
}
.gradio-container button.primary:hover {
    filter: brightness(1.08);
    box-shadow: 0 4px 12px rgba(45, 212, 191, 0.4) !important;
}
.gradio-container button.stop {
    background: #2a1215 !important;
    border: 1px solid #7f1d1d !important;
    color: #fca5a5 !important;
}
.gradio-container button.stop:hover {
    background: #3b1a1f !important;
}
.gradio-container button.secondary {
    border: 1px solid #334155 !important;
    background: #1e293b !important;
    color: #cbd5e1 !important;
}
.gradio-container button.secondary:hover {
    border-color: #2dd4bf !important;
    color: #2dd4bf !important;
    background: #14303a !important;
}

/* ── Quick sample chips ─────────────────────────────────────────────────── */
.gradio-container button.quick-chip {
    border-radius: 999px !important;
    border: 1px solid #334155 !important;
    background: #1e293b !important;
    color: #cbd5e1 !important;
    font-size: 0.8rem !important;
    padding: 0.4rem 0.85rem !important;
    box-shadow: none !important;
}
.gradio-container button.quick-chip:hover {
    border-color: #2dd4bf !important;
    color: #2dd4bf !important;
    background: #14303a !important;
}

/* ── Status: vertical step flow ─────────────────────────────────────────── */
.flow-steps {
    display: flex;
    flex-direction: column;
}
.flow-step {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.42rem 0;
    position: relative;
}
.flow-step:not(:last-child)::after {
    content: "";
    position: absolute;
    left: 10px;
    top: 100%;
    width: 2px;
    height: 14px;
    background: #334155;
}
.flow-step .dot {
    width: 22px;
    height: 22px;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 0.72rem;
    font-weight: 700;
    background: #1e293b;
    border: 1px solid #334155;
    color: #64748b;
    flex-shrink: 0;
}
.flow-step .flow-label {
    color: #94a3b8;
    font-size: 0.88rem;
}
.flow-step.done .dot {
    background: rgba(45, 212, 191, 0.16);
    border-color: #2dd4bf;
    color: #2dd4bf;
}
.flow-step.done .flow-label {
    color: #94a3b8;
}
.flow-step.active .dot {
    background: rgba(45, 212, 191, 0.24);
    border-color: #2dd4bf;
    color: #5eead4;
    box-shadow: 0 0 0 3px rgba(45, 212, 191, 0.25);
}
.flow-step.active .flow-label {
    color: #5eead4;
    font-weight: 600;
}
.flow-step.error .dot {
    background: rgba(248, 113, 113, 0.16);
    border-color: #f87171;
    color: #f87171;
}
.flow-step.error .flow-label {
    color: #f87171;
}

/* ── Status: progress card ──────────────────────────────────────────────── */
.status-card {
    background: #16213a;
    border: 1px solid #2b3a55;
    border-radius: 10px;
    padding: 0.7rem 0.95rem;
    font-size: 0.88rem;
    color: #cbd5e1;
}
.status-card .prow {
    margin: 0.3rem 0;
}
.status-card .psec {
    margin: 0.45rem 0;
}
.status-card .pitem {
    margin: 0.25rem 0 0.25rem 0.6rem;
}
.status-card .pdetail {
    display: block;
    color: #64748b;
    margin: 0.15rem 0 0 0.8rem;
    font-size: 0.8rem;
}

/* ── Chat area height ───────────────────────────────────────────────────── */
#chat-panel {
    height: calc(100vh - 300px) !important;
    min-height: 460px;
}

/* ── Fixed three-column layout ───────────────────────────────────────────── */
/* Gradio rows wrap by default; the main row must never re-stack. */
.main-row {
    flex-wrap: nowrap !important;
}
.main-row > .block {
    min-width: 0 !important;
}

/* ── Narrow windows: keep the three columns, allow horizontal scroll ─────── */
/* The fixed min-width on .gradio-container already keeps the columns side
   by side; only cosmetic tweaks remain below 800px. */
@media (max-width: 800px) {
    #app-header h1 { font-size: 1.2rem; }
}
"""


# ── Runner builder ──────────────────────────────────────────────────────────
def _build_agent_runner(
    llm_provider: str,
    materials_repo: str,
    planner_fixture: str | None,
    materials_fixture: str | None,
    run_root: Path,
    *,
    thinking_mode: bool | None = None,
) -> Any:
    from langgraph.checkpoint.sqlite import SqliteSaver

    from materials_screening.agent.conversation_store import (
        SqliteConversationStore,
    )
    from materials_screening.agent.intern_model import InternAgentModel
    from materials_screening.agent.runner import MaterialAgentRunner
    from materials_screening.agent.settings import AgentSettings, shared_intern_settings
    from materials_screening.agent_tools.result_reader import (
        FileWorkflowResultReader,
    )
    from materials_screening.llm.base import StructuredLLM
    from materials_screening.llm.factory import LLMProviderName, create_llm_provider
    from materials_screening.llm.mock_provider import MockStructuredProvider
    from materials_screening.planner.models import PlannerDraft
    from materials_screening.planner.service import PlannerService
    from materials_screening.planner.settings import (
        Settings as PlannerSettings,
    )
    from materials_screening.repositories.base import MaterialsRepository
    from materials_screening.repositories.materials_project import (
        MaterialsProjectRepository,
        map_summary_document,
    )
    from materials_screening.repositories.mock import (
        FIXED_TEST_TIME,
        MockMaterialsRepository,
    )
    from materials_screening.services.filter_service import FilterService
    from materials_screening.services.material_database_service import (
        MaterialDatabaseService,
    )
    from materials_screening.services.query_result_store import QueryResultStore
    from materials_screening.services.ranking_service import RankingService
    from materials_screening.services.validation_service import (
        ValidationService,
    )
    from materials_screening.sub_agents.materials_database.deterministic_model import (
        DeterministicDatabaseModel,
    )
    from materials_screening.sub_agents.materials_database.mock_model import (
        MaterialsDatabaseMockModel,
    )
    from materials_screening.sub_agents.materials_database.prompt import (
        SYSTEM_PROMPT as DATABASE_SYSTEM_PROMPT,
    )
    from materials_screening.sub_agents.materials_database.tools import (
        build_tool_registry as build_database_tool_registry,
    )
    from materials_screening.workflow.artifact_store import (
        FileRunArtifactStore,
    )
    from materials_screening.workflow.checkpointer import (
        create_checkpointer_handle,
    )
    from materials_screening.workflow.context import WorkflowContext
    from materials_screening.workflow.export_adapter import (
        WorkflowExportAdapter,
    )
    from materials_screening.workflow.runner import WorkflowRunner
    from materials_screening.workflow.settings import WorkflowSettings

    # Planner
    ps = PlannerSettings(llm_provider=llm_provider)
    fixtures: dict[str, Any] = {}
    err_fixtures: dict[str, Any] = {}
    if planner_fixture:
        raw = json.loads(Path(planner_fixture).read_text("utf-8"))
        f_list = raw if isinstance(raw, list) else raw.get("fixtures", [])
        e_list = [] if isinstance(raw, list) else raw.get("errors", [])
        for entry in f_list:
            fixtures[entry["query"]] = PlannerDraft.model_validate(entry["draft"])
        for entry in e_list:
            err_fixtures[entry["query"]] = entry["kind"]
    if LLMProviderName(ps.llm_provider) is LLMProviderName.MOCK:
        llm: StructuredLLM = MockStructuredProvider(
            fixtures=fixtures,
            error_fixtures=err_fixtures,
        )
    else:
        llm = create_llm_provider(ps)
    planner_service = PlannerService(settings=ps, provider=llm)

    # Repository
    if materials_repo == "mock":
        records: list[Any] = []
        if materials_fixture:
            docs = json.loads(Path(materials_fixture).read_text("utf-8"))
            for doc in docs:
                records.append(
                    map_summary_document(
                        doc,
                        database_version="fixture-v1",
                        retrieved_at=FIXED_TEST_TIME,
                    )
                )
        repository: MaterialsRepository = MockMaterialsRepository(
            records=tuple(records),
        )
    else:
        api_key = os.getenv("MP_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("MP_API_KEY 未设置")
        repository = MaterialsProjectRepository(api_key=api_key)

    # Workflow
    def _utc_now() -> datetime:
        return datetime.now(UTC)

    class _Ids:
        @staticmethod
        def new_id() -> str:
            import uuid

            return str(uuid.uuid4())

    wf_settings = WorkflowSettings()
    context = WorkflowContext(
        planner_service=planner_service,
        materials_repository=repository,
        filter_service=FilterService(),
        ranking_service=RankingService(),
        validation_service=ValidationService(),
        export_service=WorkflowExportAdapter(run_root),
        artifact_store=FileRunArtifactStore(run_root),
        clock=_utc_now,
        id_generator=_Ids(),
    )
    handle = create_checkpointer_handle(settings=wf_settings)
    wf_runner = WorkflowRunner(
        settings=wf_settings,
        context=context,
        checkpointer=handle,
    )

    # Agent model
    agent_settings = AgentSettings(
        agent_system_prompt=DATABASE_SYSTEM_PROMPT,
        agent_max_model_calls_per_turn=6,
        agent_max_tool_calls_per_turn=5,
        agent_allow_multi_step_tools=True,
        agent_max_tool_output_bytes=262_144,
        agent_max_input_bytes=524_288,
    )
    agent_model: Any
    if llm_provider == "intern":
        api_key = os.getenv("INTERN_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("INTERN_API_KEY 未设置")
        agent_model = InternAgentModel(
            shared_intern_settings(agent_settings, thinking_mode=thinking_mode),
            api_key=SecretStr(api_key),
        )
    else:
        # Offline demo mode: the mock model drives the real (mock) workflow so
        # the UI answers with genuine screening results, never a canned reply.
        agent_model = MaterialsDatabaseMockModel()

    if llm_provider == "intern":
        agent_model = DeterministicDatabaseModel(agent_model)

    tool_registry = build_database_tool_registry(
        MaterialDatabaseService(
            repository,
            QueryResultStore(Path("data/material_queries")),
        )
    )

    store = SqliteConversationStore(Path("data/agent_conversations.sqlite"))
    cp_path = agent_settings.agent_checkpoint_db
    cp_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(cp_path), check_same_thread=False)
    saver = SqliteSaver(conn)

    runner = MaterialAgentRunner(
        settings=agent_settings,
        store=store,
        workflow_runner=wf_runner,
        workflow_result_reader=FileWorkflowResultReader(run_root),
        tool_registry=tool_registry,
        agent_model=agent_model,
        checkpointer=saver,
    )
    return runner


# ── Flowchart (vertical step flow) ─────────────────────────────────────────
def _flowchart(active: list[str], done: list[str], err: str | None) -> str:
    order = [
        ("prepare_turn", "📝 准备对话"),
        ("call_agent_model", "🤖 调用 AI 模型"),
        ("execute_tools", "⚡ 执行工具"),
        ("validate_final", "✅ 验证回答"),
        ("finalize_success", "🎉 完成"),
        ("finalize_error", "⚠️ 错误"),
    ]
    rows = []
    for key, label in order:
        if key == err:
            state, dot = "error", "✕"
        elif key in active:
            state, dot = "active", "●"
        elif key in done:
            state, dot = "done", "✓"
        else:
            state, dot = "", ""
        rows.append(
            f'<div class="flow-step {state}">'
            f'<span class="dot">{dot}</span>'
            f'<span class="flow-label">{label}</span>'
            f"</div>"
        )
    return '<div class="flow-steps">' + "".join(rows) + "</div>"


# ── Progress details (card layout) ─────────────────────────────────────────
def _progress(
    tools: list[str],
    mid: list[dict[str, Any]],
    tid: str | None,
    conversation: str | None = None,
) -> str:
    sections: list[str] = []
    if conversation:
        sections.append(f'<div class="prow">💬 会话: {conversation}</div>')
    if tools:
        tool_items = "".join(f'<div class="pitem">▸ {t}</div>' for t in tools)
        sections.append(f'<div class="psec"><b>🔧 工具调用</b>{tool_items}</div>')
    if mid:
        mid_items: list[str] = []
        for item in mid:
            detail = (
                f'<small class="pdetail">{item["detail"]}</small>'
                if item.get("detail")
                else ""
            )
            mid_items.append(
                f'<div class="pitem">{item.get("icon", "")} '
                f"{item.get('label', '')}{detail}</div>"
            )
        sections.append(
            f'<div class="psec"><b>📊 中间结果</b>{"".join(mid_items)}</div>'
        )
    if tid:
        sections.append(f'<div class="prow"><small>thread: {tid}</small></div>')
    if not sections:
        sections.append('<div class="prow"><i>等待执行…</i></div>')
    return '<div class="status-card">' + "".join(sections) + "</div>"


# ── Result reader ───────────────────────────────────────────────────────────
def _read_result(tid: str | None) -> dict[str, Any] | None:
    if not tid:
        return None
    p = _run_root / tid / "result.json"
    if not p.exists():
        return None
    try:
        payload = json.loads(p.read_text("utf-8"))
        return payload if isinstance(payload, dict) else None
    except (json.JSONDecodeError, OSError):
        return None


# ── Error explanations (user-facing Chinese) ────────────────────────────────
# Agent errors keep stable English codes; this table turns them into plain
# Chinese "what happened + what to do" guidance shown in the chat. The raw
# code stays visible at the end for debugging and bug reports.
_ERROR_EXPLAIN: dict[str, tuple[str, str, str]] = {
    "MODEL_ERROR": (
        "模型输出异常",
        "AI 模型没有返回规范格式的回答，通常是网络波动、限流或输出被截断导致。",
        "稍后重试；若反复出现，可在左侧把「LLM 提供商」切到 mock 离线模式，"
        "或检查 `.env` 中的 INTERN_API_KEY。",
    ),
    "MODEL_INCOMPLETE": (
        "模型回答被截断",
        "AI 模型的回答超过了长度上限，被中途截断。",
        "把问题改得简洁一些（例如减少要求的候选数量）后重试。",
    ),
    "MODEL_EMPTY_OUTPUT": (
        "模型未返回内容",
        "AI 模型返回了空回答。",
        "重试一次；若反复出现，可切换到 mock 离线模式。",
    ),
    "MODEL_CALL_LIMIT": (
        "对话步骤过多",
        "本轮对话的模型调用次数达到上限，未能完成回答。",
        "重新提问一个更直接的问题，或点击「🔄 重置」开始新会话。",
    ),
    "INPUT_TOO_LARGE": (
        "内容超出长度限制",
        "对话内容过长，超出了允许的输入上限。",
        "点击「🔄 重置」开始新会话，或精简问题后重试。",
    ),
    "INVALID_FINAL_DRAFT": (
        "模型回答格式错误",
        "AI 模型返回的回答不符合要求的结构。",
        "重试一次；若反复出现，可切换到 mock 离线模式。",
    ),
    "FINAL_VALIDATION_FAILED": (
        "回答未通过安全检查",
        "AI 模型的回答未能通过事实与安全校验（例如引用了不存在的材料或证据）。",
        "换个问法重试；此问题不影响已完成的筛选结果。",
    ),
    "TURN_LIMIT": (
        "会话已达上限",
        "当前会话的对话轮数已用尽。",
        "点击「🔄 重置」开始新会话。",
    ),
    "CONVERSATION_BUSY": (
        "上一条消息仍在处理",
        "同一会话同时只能处理一条消息。",
        "等待上一条完成，或点击「⏹️ 停止」。",
    ),
    "RECURSION_LIMIT": (
        "对话循环过深",
        "Agent 的推理步骤超出限制，本轮未完成。",
        "重新提问一个更直接的问题。",
    ),
    "EMPTY_MESSAGE": (
        "消息为空",
        "没有收到要处理的内容。",
        "输入内容后重试。",
    ),
}

_UNKNOWN_ERROR_EXPLAIN = (
    "处理失败",
    "发生了未预期的错误。",
    "可重试；若反复出现，请把界面末尾的错误码反馈给维护者。",
)


def _explain_error(
    code: str,
    message: str,
    thread_id: str | None = None,
) -> str:
    """Build plain-Chinese guidance for an agent error code.

    The stable English error code and raw message stay visible at the end so
    bugs can be reported; when the screening workflow still produced a result
    thread, that is stated honestly — the failure is the answer generation,
    not the screening itself.
    """
    title, detail, advice = _ERROR_EXPLAIN.get(code, _UNKNOWN_ERROR_EXPLAIN)
    parts = [f"❌ **{title}**", detail, f"💡 建议：{advice}"]
    if thread_id:
        parts.append(
            f"📁 说明：本次筛选工作流已执行完成，结果保存在 "
            f"`data/workflow_runs/{thread_id}`，可用导出按钮保存；"
            "仅对话回答生成失败。"
        )
    parts.append(f"*错误码：`{code}` — {message}*")
    return "\n\n".join(parts)


# ── Chat handler (generator) ────────────────────────────────────────────────
def _chat_respond(
    message: str,
    history: list[dict[str, Any]],
) -> Generator[tuple[Any, ...], None, None]:
    global _conversation_id, _cancel_event, _turn_in_progress
    if not message.strip():
        yield (
            history,
            _flowchart([], [], None),
            "",
            "",
        )
        return

    if _global_runner is None:
        history.append({"role": "user", "content": message})
        history.append(
            {
                "role": "assistant",
                "content": "⚠️ 请先在左侧面板初始化 Agent",
            }
        )
        yield (
            history,
            _flowchart([], [], None),
            "",
            "",
        )
        return

    if _turn_in_progress:
        history.append({"role": "user", "content": message})
        history.append(
            {
                "role": "assistant",
                "content": "⚠️ 上一条消息仍在处理中，请稍候或点击 ⏹️ 停止",
            }
        )
        yield (
            history,
            _flowchart([], [], None),
            _progress([], [], None, _conversation_id),
            "",
        )
        return

    _turn_in_progress = True
    _cancel_event = threading.Event()
    history.append({"role": "user", "content": message})
    executed: list[str] = []
    tools: list[str] = []
    final: Any = None

    # Yield immediately so UI shows "starting" state, not blank
    yield (
        history + [{"role": "assistant", "content": "⏳ 正在处理…"}],
        _flowchart([], [], None),
        _progress([], [], None, _conversation_id),
        "",
    )

    model_calls = 0
    try:
        for event in _global_runner.ask_stream(
            message=message,
            conversation_id=_conversation_id,
            cancel_event=_cancel_event,
        ):
            if event.is_final:
                final = event.result
            else:
                if event.node not in executed:
                    executed.append(event.node)
                if event.node == "call_agent_model":
                    model_calls += 1
                if "工具" in event.message:
                    tools.append(event.message)
                done = executed[:-1] if len(executed) > 1 else []
                cur = [executed[-1]] if executed else []
                # Show model call count in progress
                pg = _progress(
                    tools,
                    [],
                    f"书生 API 第 {model_calls} 次调用完成（共约 3 次）"
                    if model_calls
                    else None,
                    _conversation_id,
                )
                tip = ""
                if event.node == "call_agent_model":
                    tip = "（正在等待书生 API 响应）"
                yield (
                    history + [{"role": "assistant", "content": f"⏳ 处理中…{tip}"}],
                    _flowchart(cur, done, None),
                    pg,
                    "",
                )
    finally:
        _turn_in_progress = False
        _cancel_event = None

    if final is None:
        history.append({"role": "assistant", "content": "⏹️ 已停止"})
        yield (
            history,
            _flowchart([], executed, None),
            "",
            "",
        )
        return

    # Keep the same conversation across messages so follow-ups work.
    _conversation_id = final.conversation_id

    tid = final.active_workflow_thread_id
    resp = final.response_text
    if final.error:
        resp = _explain_error(
            final.error.get("code", "ERROR"),
            final.error.get("message", ""),
            tid,
        )
    if final.status == "cancelled":
        resp = f"⏹️ {resp}"

    history.append({"role": "assistant", "content": resp})

    mid: list[dict[str, Any]] = []
    if tid:
        wf = _read_result(tid)
        if wf:
            mid.append(
                {
                    "icon": "🔍",
                    "label": "工作流执行完成",
                    "detail": (
                        f"检索 {wf.get('retrieved_count', 0)} → "
                        f"过滤 {wf.get('passed_filter_count', 0)} → "
                        f"返回 {wf.get('returned_count', 0)}"
                    ),
                }
            )

    yield (
        history,
        _flowchart([], executed, None),
        _progress(tools, mid, tid, _conversation_id),
        "",
    )

    _saved_results.append(
        {
            "ts": datetime.now(UTC).isoformat(),
            "query": message,
            "status": final.status if final else "error",
            "response": final.response_text if final else "",
            "thread_id": tid or "",
        }
    )


# ── Event handlers ──────────────────────────────────────────────────────────
def _on_init(provider: str, repo: str, pf: str, mf: str) -> tuple[str, str]:
    global _global_runner, _conversation_id
    try:
        _global_runner = _build_agent_runner(
            provider,
            repo,
            pf.strip() or None,
            mf.strip() or None,
            _run_root,
        )
        # A fresh runner owns a fresh conversation; the previous one is gone.
        _conversation_id = None
        if provider == "intern":
            note = (
                "✅ **已就绪** (Intern 35B)  |  "
                "每条消息约需 **3 次** API 调用（Agent→工作流(含解析)→Agent），"
                "总计约 **10-30 秒**，请耐心等待"
            )
        else:
            note = (
                "✅ **已就绪**（离线演示模式 mock）|  "
                "使用内置示例数据执行完整筛选流程，不调用真实 API；"
                "要使用真实模型请选择 **intern** 并在 `.env` 配置 "
                "`INTERN_API_KEY`"
            )
        return note, _flowchart([], [], None)
    except Exception as exc:
        return f"❌ {exc}", _flowchart([], [], None)


def _on_stop() -> str | None:
    """Set the cooperative cancel event; the turn ends at the next boundary."""
    global _cancel_event
    event = _cancel_event
    if event is None:
        return "<i>当前没有正在进行的任务</i>"
    event.set()
    return "<b>⏹️ 停止请求已发送（当前步骤完成后停止，不再继续后续步骤）</b>"


def _on_reset() -> tuple[Any, ...]:
    global _conversation_id
    _conversation_id = None
    return (
        [],
        _flowchart([], [], None),
        _progress([], [], None, None) + "<br><b>🔄 已重置，开始新会话</b>",
        "",
    )


def _on_export_json() -> str | None:
    if not _saved_results:
        return None
    r = _saved_results[-1]
    p = Path("data/exports/web_export.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(p)


def _on_export_csv() -> str | None:
    if not _saved_results:
        return None
    r = _saved_results[-1]
    lines = [
        "timestamp,query,status,response,thread_id",
        f'"{r["ts"]}","{r["query"]}","{r["status"]}",'
        f'"{r["response"]}","{r["thread_id"]}"',
    ]
    p = Path("data/exports/web_export.csv")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines), encoding="utf-8")
    return str(p)


def _on_copy(history: list[dict[str, Any]]) -> str:
    if not history:
        return ""
    for msg in reversed(history):
        if msg["role"] == "assistant":
            return str(msg["content"])
    return ""


# ── UI ──────────────────────────────────────────────────────────────────────
def _create_ui() -> gr.Blocks:
    qa_prompts = _load_qa_prompts() or QA_PROMPTS
    sample = random.sample(qa_prompts, min(10, len(qa_prompts)))

    demo: gr.Blocks = gr.Blocks(title="材料筛选系统", elem_id="app-root")
    with demo:
        gr.Markdown(
            "# 🔬 材料筛选系统\n\n确定性无机半导体筛选 · 自然语言 Agent 演示",
            elem_id="app-header",
        )

        # Fixed three-column layout: the row never wraps, so the left/middle/
        # right panels always stay side by side (narrow windows scroll
        # horizontally instead of re-stacking).
        with gr.Row(elem_classes=["main-row"]):
            # ── LEFT: Config ──────────────────────────────────────────
            with gr.Column(scale=1):
                with gr.Accordion("⚙️ 配置", open=True):
                    provider = gr.Dropdown(
                        ["mock", "intern"],
                        value="mock",
                        label="LLM 提供商",
                    )
                    repo = gr.Dropdown(
                        ["mock", "materials-project"],
                        value="mock",
                        label="数据源",
                    )
                    with gr.Row():
                        init_btn = gr.Button(
                            "🔌 初始化",
                            variant="primary",
                        )
                        reset_btn = gr.Button("🔄 重置", size="sm")
                    status_md = gr.Markdown("未初始化")

                with gr.Accordion("🔧 高级选项", open=False):
                    pf = gr.Textbox(
                        value="tests/fixtures/planner_cli_fixtures.json",
                        label="Planner Fixture",
                    )
                    mf = gr.Textbox(
                        value="tests/fixtures/mp_documents.json",
                        label="Materials Fixture",
                    )

                with gr.Accordion("💡 快捷示例", open=False):
                    quick_btns: list[tuple[gr.Button, str]] = []
                    for p in sample:
                        btn = gr.Button(
                            p[:38] + ("…" if len(p) > 38 else ""),
                            size="sm",
                            elem_classes=["quick-chip"],
                        )
                        quick_btns.append((btn, p))

            # ── CENTER: Chat ──────────────────────────────────────────
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(
                    label="🤖 材料筛选助手",
                    height=560,
                    elem_id="chat-panel",
                )
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="输入你的材料筛选需求…",
                        label="",
                        scale=5,
                    )
                    send_btn = gr.Button(
                        "🚀 发送",
                        variant="primary",
                        scale=1,
                    )
                    stop_btn = gr.Button(
                        "⏹️ 停止",
                        variant="stop",
                        scale=1,
                    )
                with gr.Row():
                    dl_json = gr.Button("📥 JSON", size="sm")
                    dl_csv = gr.Button("📥 CSV", size="sm")
                    cp_btn = gr.Button("📋 复制", size="sm")
                    dl_out = gr.File(label="", visible=False)

            # ── RIGHT: Status ─────────────────────────────────────────
            with gr.Column(scale=2):
                gr.Markdown("### 📈 执行流程")
                flowchart = gr.HTML(_flowchart([], [], None))
                gr.Markdown("### 📋 进度与结果")
                progress = gr.HTML(_progress([], [], None))

        # ── Events ────────────────────────────────────────────────────
        init_btn.click(
            _on_init,
            [provider, repo, pf, mf],
            [status_md, flowchart],
        )
        reset_btn.click(
            _on_reset,
            None,
            [chatbot, flowchart, progress, msg],
        )

        send_btn.click(
            _chat_respond,
            [msg, chatbot],
            [chatbot, flowchart, progress, msg],
        )
        msg.submit(
            _chat_respond,
            [msg, chatbot],
            [chatbot, flowchart, progress, msg],
        )
        # Cooperative stop: set the cancel event instead of cancelling the
        # generator, so the turn ends cleanly at the next node boundary and
        # the conversation state stays consistent for follow-up messages.
        stop_btn.click(_on_stop, None, [progress])

        dl_json.click(_on_export_json, None, [dl_out])
        dl_csv.click(_on_export_csv, None, [dl_out])
        cp_btn.click(_on_copy, [chatbot], [msg])

        for btn, txt in quick_btns:
            btn.click(
                lambda t=txt: t,
                None,
                [msg],
            ).then(
                _chat_respond,
                [msg, chatbot],
                [chatbot, flowchart, progress, msg],
            )

    return demo


def main() -> None:
    port = int(os.environ.get("GRADIO_SERVER_PORT", "8501"))
    _create_ui().launch(
        server_name="127.0.0.1",
        server_port=port,
        share=False,
        css=CSS,
        theme=gr.themes.Soft(
            primary_hue=gr.themes.colors.teal,
            secondary_hue=gr.themes.colors.blue,
            neutral_hue=gr.themes.colors.slate,
            radius_size=gr.themes.sizes.radius_lg,
            font=[
                "Segoe UI",
                "Microsoft YaHei",
                "PingFang SC",
                "Helvetica Neue",
                "Arial",
                "sans-serif",
            ],
        ),
    )


if __name__ == "__main__":
    main()
