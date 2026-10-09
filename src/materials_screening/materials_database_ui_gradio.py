"""Dedicated Gradio UI for the Materials Database Agent.

Launch with ``uv run materials-screen master db-ui`` or run this module.
The page talks to the real ``materials_database`` runner; UI state only stores
the active conversation id and never fabricates query results.
"""

from __future__ import annotations

import html
import os
import re
import threading
from collections.abc import Generator
from pathlib import Path
from typing import Any

import gradio as gr
from dotenv import load_dotenv

from materials_screening.agent_ui_gradio import _build_agent_runner

load_dotenv()

_runner: Any = None
_runner_lock = threading.Lock()

QUICK_PROMPTS = (
    (
        "条件筛选",
        "筛选含 Li、Fe、O，带隙大于 1 eV，凸包上能量不超过 0.05 "
        "eV/atom 的稳定材料，返回材料 ID、化学式、带隙和凸包上能量",
    ),
    (
        "排序 Top-K",
        "查询含 Si 和 O 的稳定材料，按 band_gap_ev 从高到低排序，返回前 10 个",
    ),
    (
        "材料详情",
        "查询 mp-149 的材料详情，返回化学式、元素、带隙、密度、形成能、"
        "凸包上能量、稳定性、晶系和空间群",
    ),
    (
        "多材料比较",
        "比较 mp-149、mp-13 和 mp-2534 的带隙、密度、形成能和凸包上能量",
    ),
    ("描述统计", "统计刚才查询结果中 band_gap_ev 和 density_g_cm3 的分布"),
    (
        "离群检测",
        "对刚才的查询结果执行 band_gap_ev 离群检测，列出离群材料、检测方法、"
        "阈值和样本数",
    ),
    ("结果导出", "把刚才查询快照中的完整结构化材料结果导出为 CSV"),
)

TOOL_NAMES = {
    "search_materials": "查询、筛选、排序与 Top-K",
    "get_material_details": "材料详情",
    "get_query_result": "查询快照分页",
    "compare_materials": "多材料比较",
    "describe_materials": "描述统计",
    "detect_material_outliers": "离群检测",
    "export_materials": "结果导出",
}

_SUBSCRIPT_DIGITS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
_FORMULA_HEADERS = {"化学式", "formula", "formula_pretty", "formula pretty"}
_LABELED_FORMULA_PATTERN = re.compile(
    r"(?P<label>(?:化学式|formula(?:_pretty)?)[ \t]*[:：][ \t]*)"
    r"(?P<formula>[A-Z][A-Za-z0-9()\[\].·+-]*)",
    re.IGNORECASE,
)


def _formula_with_subscripts(formula: str) -> str:
    """Render stoichiometric digits as Unicode subscripts for the chat UI."""
    return formula.translate(_SUBSCRIPT_DIGITS)


def _format_chemical_formulas(text: str) -> str:
    """Format only explicit formula fields and formula columns in Markdown."""
    lines = text.splitlines(keepends=True)
    formula_columns: set[int] = set()
    rendered: list[str] = []
    for line in lines:
        newline = "\n" if line.endswith("\n") else ""
        content = line[:-1] if newline else line
        if "|" in content:
            cells = content.split("|")
            header_columns = {
                index
                for index, cell in enumerate(cells)
                if cell.strip().casefold() in _FORMULA_HEADERS
            }
            if header_columns:
                formula_columns = header_columns
            if formula_columns:
                for index in formula_columns:
                    if index < len(cells):
                        cells[index] = _formula_with_subscripts(cells[index])
                content = "|".join(cells)
        else:
            formula_columns = set()
        content = _LABELED_FORMULA_PATTERN.sub(
            lambda match: (
                match.group("label") + _formula_with_subscripts(match.group("formula"))
            ),
            content,
        )
        rendered.append(content + newline)
    return "".join(rendered)


CSS = """
footer { display: none !important; }
body, .gradio-container { background: #f4f7fb !important; }
.gradio-container {
    max-width: 1560px !important;
    margin: 0 auto !important;
    padding: 12px !important;
    font-family: "Segoe UI", "Microsoft YaHei", sans-serif !important;
}
#db-header {
    background: linear-gradient(120deg, #12233f, #1c3d68);
    color: #fff;
    border-radius: 14px;
    padding: 16px 20px;
    margin-bottom: 12px;
    box-shadow: 0 12px 30px rgba(18, 35, 63, .18);
}
#db-header h1 { margin: 0 0 4px; font-size: 21px; }
#db-header p { margin: 0; color: #c8d8ee; font-size: 13px; }
.db-main { align-items: stretch !important; gap: 12px !important; }
.db-panel {
    background: #fff !important;
    border: 1px solid #dce4ef !important;
    border-radius: 14px !important;
    padding: 12px !important;
    box-shadow: 0 5px 18px rgba(20, 38, 66, .06) !important;
}
.db-panel .block { border-radius: 11px !important; }
#db-chat { min-height: 430px !important; }

/* ── Chat bubbles (Gradio 6 DOM: .message-row.bubble.user-row / .bot-row) ── */
.bubble-wrap .message-row.bubble.user-row { justify-content: flex-end; }
.bubble-wrap .message-row.bubble.bot-row { justify-content: flex-start; }
.bubble-wrap .user.message {
    background: #2347b8 !important;
    border: none !important;
    border-radius: 12px 12px 4px 12px !important;
}
/* 用户气泡内的所有文字强制白色（覆盖 markdown 渲染出的 p/span/prose 默认深色） */
.bubble-wrap .user.message,
.bubble-wrap .user.message * {
    color: #ffffff !important;
}
.bubble-wrap .bot.message {
    background: #eef3fa !important;
    color: #17243a !important;
    border: 1px solid #dce4ef !important;
    border-radius: 12px 12px 12px 4px !important;
    width: 100% !important;
    max-width: 96% !important;
}

/* ── Keep markdown tables on one line per cell; scroll horizontally ──────── */
#db-chat table {
    display: block;
    max-width: 100%;
    overflow-x: auto;
    border-collapse: collapse;
}
#db-chat table th,
#db-chat table td {
    white-space: nowrap;
    padding: 6px 10px !important;
}
.quick-chip {
    border-radius: 999px !important;
    border: 1px solid #cfdaea !important;
    background: #f7f9fc !important;
    color: #34445e !important;
    font-size: 12px !important;
}
.quick-chip:hover { border-color: #3157d5 !important; color: #3157d5 !important; }
.db-capabilities { color: #4c5d75; font-size: 13px; line-height: 1.75; }
.db-capabilities strong { color: #17243a; }
.trace-card {
    border: 1px solid #dce4ef;
    border-radius: 12px;
    padding: 12px 14px;
    color: #465874;
    background: #f8fafc;
    font-size: 13px;
}
.trace-row { position: relative; padding: 0 0 13px 22px; }
.trace-row:last-child { padding-bottom: 0; }
.trace-row::before {
    content: ""; position: absolute; left: 2px; top: 5px;
    width: 9px; height: 9px; border-radius: 50%; background: #20a274;
}
.trace-row:not(:last-child)::after {
    content: ""; position: absolute; left: 6px; top: 16px;
    bottom: 2px; width: 1px; background: #cbd6e5;
}
.trace-row b { color: #17243a; font-weight: 600; }
.trace-meta { margin-top: 4px; color: #718096; font-size: 12px; }
.conversation-card {
    padding: 10px 12px; border-radius: 10px; background: #edf3ff;
    color: #2749b2; font-family: Consolas, monospace; overflow-wrap: anywhere;
}
button.primary { background: #3157d5 !important; border-color: #3157d5 !important; }
@media (max-width: 900px) {
    .db-main { flex-wrap: wrap !important; }
    .db-panel { min-width: 100% !important; }
}
"""


def _trace(
    *,
    conversation_id: str | None,
    status: str = "等待提问",
    tools: tuple[str, ...] = (),
    evidence_ids: tuple[str, ...] = (),
    event_message: str = "",
) -> str:
    tool_text = (
        "、".join(
            f"{html.escape(name)}（{html.escape(TOOL_NAMES.get(name, name))}）"
            for name in tools
        )
        or "尚未调用"
    )
    evidence = "、".join(html.escape(item) for item in evidence_ids) or "待生成"
    conversation = html.escape(conversation_id or "首次发送后自动创建")
    detail = html.escape(event_message) if event_message else "等待数据库 Agent 执行"
    return (
        '<div class="trace-card">'
        f'<div class="trace-row"><b>Conversation ID</b>'
        f'<div class="trace-meta">{conversation}</div></div>'
        f'<div class="trace-row"><b>当前状态</b>'
        f'<div class="trace-meta">{html.escape(status)} · {detail}</div></div>'
        f'<div class="trace-row"><b>数据库工具</b>'
        f'<div class="trace-meta">{tool_text}</div></div>'
        f'<div class="trace-row"><b>证据校验</b>'
        f'<div class="trace-meta">evidence_id: {evidence}</div></div>'
        "</div>"
    )


def _initialize(provider: str, repository: str) -> tuple[str, str, None]:
    global _runner
    try:
        with _runner_lock:
            _runner = _build_agent_runner(
                provider,
                repository,
                "tests/fixtures/planner_cli_fixtures.json"
                if provider == "mock"
                else None,
                "tests/fixtures/mp_documents.json" if repository == "mock" else None,
                Path("data/workflow_runs"),
                # Keep thinking mode enabled (aligned with .env's
                # INTERN_THINKING_MODE=true). Disabling it made Intern 35B
                # intermittently return an empty final message instead of
                # emitting a tool call or the AgentFinalDraft envelope.
                thinking_mode=True,
            )
        mode = (
            "Intern 35B + Materials Project"
            if provider == "intern" and repository == "materials-project"
            else f"{provider} + {repository}"
        )
        return (
            f"✅ **数据库 Agent 已就绪** · {mode}",
            _trace(conversation_id=None, status="已初始化"),
            None,
        )
    except Exception as exc:
        return (
            f"❌ 初始化失败：{exc}",
            _trace(conversation_id=None, status="初始化失败", event_message=str(exc)),
            None,
        )


def _ask(
    message: str,
    history: list[dict[str, Any]],
    conversation_id: str | None,
) -> Generator[tuple[list[dict[str, Any]], str, str, str | None], None, None]:
    if not message.strip():
        yield history, _trace(conversation_id=conversation_id), "", conversation_id
        return
    if _runner is None:
        history.append({"role": "assistant", "content": "请先点击左侧“初始化 Agent”。"})
        yield history, _trace(conversation_id=None, status="未初始化"), "", None
        return

    history.append({"role": "user", "content": message})
    yield (
        history + [{"role": "assistant", "content": "正在调用数据库 Agent…"}],
        _trace(conversation_id=conversation_id, status="运行中"),
        "",
        conversation_id,
    )

    final: Any = None
    last_event = ""
    for event in _runner.ask_stream(
        message=message,
        conversation_id=conversation_id,
    ):
        if event.is_final:
            final = event.result
            continue
        last_event = event.message
        yield (
            history + [{"role": "assistant", "content": "正在处理，请稍候…"}],
            _trace(
                conversation_id=conversation_id,
                status="运行中",
                event_message=last_event,
            ),
            "",
            conversation_id,
        )

    if final is None:
        history.append({"role": "assistant", "content": "本轮没有生成最终结果。"})
        yield (
            history,
            _trace(conversation_id=conversation_id, status="未完成"),
            "",
            conversation_id,
        )
        return

    conversation_id = final.conversation_id
    response = final.response_text
    if final.error:
        response = f"处理失败：{final.error.get('message', response)}"
    else:
        response = _format_chemical_formulas(response)
    history.append({"role": "assistant", "content": response})
    yield (
        history,
        _trace(
            conversation_id=conversation_id,
            status="完成" if final.status == "completed" else final.status,
            tools=tuple(final.selected_tools),
            evidence_ids=tuple(final.evidence_ids),
            event_message="最终回答已通过验证" if not final.error else "本轮执行失败",
        ),
        "",
        conversation_id,
    )


def _new_conversation() -> tuple[list[Any], str, None, str]:
    return [], _trace(conversation_id=None, status="新会话"), None, ""


def _create_ui() -> gr.Blocks:
    demo = gr.Blocks(title="Materials Database Agent")
    with demo:
        gr.HTML(
            "<div id='db-header'><h1>Materials Database Agent</h1>"
            "<p>自然语言驱动的 Materials Project 查询、比较、统计、"
            "离群检测与结果导出</p></div>"
        )
        conversation_state = gr.State(value=None)

        with gr.Row(elem_classes=["db-main"]):
            with gr.Column(scale=2, min_width=235, elem_classes=["db-panel"]):
                gr.Markdown("### 运行配置")
                provider = gr.Dropdown(
                    choices=["intern", "mock"], value="intern", label="基座模型"
                )
                repository = gr.Dropdown(
                    choices=["materials-project", "mock"],
                    value="materials-project",
                    label="材料数据库",
                )
                init_button = gr.Button("初始化 Agent", variant="primary")
                new_button = gr.Button("新建对话")
                init_status = gr.Markdown("尚未初始化")
                gr.Markdown(
                    """
<div class="db-capabilities">
<strong>统一能力范围</strong><br>
① 条件筛选　② 排序与 Top-K<br>
③ 材料详情　④ 多材料比较<br>
⑤ 描述统计　⑥ 离群检测<br>
⑦ CSV / JSON 导出
</div>
"""
                )

            with gr.Column(scale=6, min_width=560, elem_classes=["db-panel"]):
                chatbot = gr.Chatbot(
                    label="数据库查询对话",
                    height=430,
                    elem_id="db-chat",
                )
                with gr.Row():
                    message = gr.Textbox(
                        label="",
                        placeholder="输入材料查询；后续可直接说“统计刚才的结果”",
                        scale=6,
                    )
                    send_button = gr.Button("发送", variant="primary", scale=1)
                gr.Markdown("**功能演示命令**")
                quick_buttons: list[tuple[gr.Button, str]] = []
                with gr.Row():
                    for label, prompt in QUICK_PROMPTS[:4]:
                        quick_buttons.append(
                            (
                                gr.Button(
                                    label, size="sm", elem_classes=["quick-chip"]
                                ),
                                prompt,
                            )
                        )
                with gr.Row():
                    for label, prompt in QUICK_PROMPTS[4:]:
                        quick_buttons.append(
                            (
                                gr.Button(
                                    label, size="sm", elem_classes=["quick-chip"]
                                ),
                                prompt,
                            )
                        )

            with gr.Column(scale=2, min_width=260, elem_classes=["db-panel"]):
                gr.Markdown("### 子 Agent 执行轨迹")
                trace = gr.HTML(_trace(conversation_id=None))
                gr.Markdown(
                    """
### 工作原理

自然语言由 **Intern 35B** 转换为受 Schema 约束的工具参数；确定性服务访问
**Materials Project**。首次查询保存 `query_id`，相同 Conversation ID 的后续
统计、离群检测和导出会复用同一查询快照。最终回答必须通过证据校验。
"""
                )

        init_button.click(
            _initialize,
            [provider, repository],
            [init_status, trace, conversation_state],
        )
        new_button.click(
            _new_conversation,
            None,
            [chatbot, trace, conversation_state, message],
        )
        send_button.click(
            _ask,
            [message, chatbot, conversation_state],
            [chatbot, trace, message, conversation_state],
        )
        message.submit(
            _ask,
            [message, chatbot, conversation_state],
            [chatbot, trace, message, conversation_state],
        )
        for button, prompt in quick_buttons:
            button.click(lambda value=prompt: value, None, message)

    return demo


def main() -> None:
    port = int(os.getenv("GRADIO_SERVER_PORT", "8502"))
    _create_ui().launch(
        server_name="127.0.0.1",
        server_port=port,
        share=False,
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
        css=CSS,
    )


if __name__ == "__main__":
    main()
