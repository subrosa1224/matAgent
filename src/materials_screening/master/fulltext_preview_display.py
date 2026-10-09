"""Readable preview only; stored quotes and source checks stay untouched."""

import html
import re

_CONTROLS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ufffd]")
_ESCAPED_CONTROLS = re.compile(r"\\u(?:000[0-8bcef]|001[0-9a-f]|007f|fffd)", re.I)
_MARKER = "[符号识别待核对]"
_ARRANGEMENTS = {
    "hold": "等待确认",
    "extract": "已选定待详细分析",
    "background": "仅作背景阅读",
    "exclude": "暂不纳入分析",
    "clarify": "需核对研究对象",
    "尚未决定": "待确认",
}


def preview_report_lines(
    task, selected, validated_documents, *, awaiting_confirmation=False, reused=0
):
    """Build raw report from checked task data; never alter a stored preview."""
    previews = {preview.document_id: preview for preview in task.previews}
    lines = ["全文预览（仅用于选文，不是实验统计或专业验证）："]
    if selected:
        completed = sum(
            task.document_ids[index] in validated_documents
            and task.document_ids[index] in previews
            for index in selected
        )
        summary = f"本次选定 {len(selected)} 篇；已预览 {completed}/{len(selected)} 篇"
        if completed == len(selected) and awaiting_confirmation:
            summary += "，等待你确认分析范围"
        lines.append(summary + "。")
    for index in selected:
        doc = task.document_ids[index]
        if doc not in validated_documents or doc not in previews:
            continue
        preview = previews[doc]
        decision = task.preview_decisions.get(doc)
        lines.extend(
            (
                f"\n第{index + 1}篇：{preview.title}",
                f"研究问题：{preview.research_question}",
                f"方法：{preview.methods}",
                f"主要发现（预览）：{preview.key_findings}",
                f"原文依据（第{preview.page_from}页）：{preview.evidence_quote}",
                f"阶段决定：{decision.action if decision else '尚未决定'}；"
                f"{decision.reason if decision else ''}",
            )
        )
    if reused:
        lines.append(
            f"\n复用并重新核对了 {reused} 篇当前任务的已保存预览；"
            "这些预览未重新请求模型。"
        )
    return lines


def render_fulltext_preview(answer: str) -> str:
    # The raw response already contains authoritative selected/validated counts.
    # Never infer a completed denominator from displayed ordinals or cache hits.
    main = answer.replace(
        "全文预览（仅用于选文，不是实验统计或专业验证）：",
        "论文预览结果（仅用于选文，不是实验统计或专业验证）：",
        1,
    )
    main = re.sub(
        r"(?m)^复用并重新核对了[^\n]*(?:\n这些预览未重新请求模型。)?\n?",
        "",
        main,
    )

    def arrangement(match):
        label = _ARRANGEMENTS.get(match[1], "待确认")
        return "阅读安排：" + label + "；" + match[2]

    main = re.sub(r"(?m)^阶段决定：([^；\n]+)；([^\n]*)", arrangement, main)
    has_bad_symbols = bool(_CONTROLS.search(main) or _ESCAPED_CONTROLS.search(main))
    main = _ESCAPED_CONTROLS.sub(_MARKER, _CONTROLS.sub(_MARKER, main))
    if has_bad_symbols:
        main = re.sub(
            r"原文依据（第(\d+)页）：",
            r"原文摘录（第\1页；异常符号已标注）：",
            main,
        )
        main += (
            "\n\n说明：部分 PDF 符号提取异常，已标注为“符号识别待核对”。"
            "这些位置不作为已确认的单位、公式或数值，原始引文保留在折叠详情中。"
        )
    if "[数值待深度分析]" in main:
        main += "\n\n“数值待深度分析”表示该数值尚未核验，并非已确认实验数据。"
    audit = _CONTROLS.sub(lambda match: f"\\u{ord(match[0]):04x}", answer)
    return (
        html.escape(main.strip())
        + "\n\n<details>\n<summary>原始预览与处理详情</summary>\n\n<pre>"
        + html.escape(audit)
        + "</pre>\n</details>"
    )
