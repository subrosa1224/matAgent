"""User-visible candidate progress from saved tool evidence, never model claims."""

from .models import LiteratureSearchOutput
from .service import _paper_mentions_formula
from .topic_scope import explicit_candidate_formulas


def render_search_coverage(output: dict) -> str:
    try:
        result = LiteratureSearchOutput.model_validate(output)
    except ValueError:
        return ""
    if result.expanded_query is None:
        return ""
    candidates = explicit_candidate_formulas(result.expanded_query.original_topic)
    if not candidates:
        return ""
    lines = [
        "**各体系检索进度**",
        "",
        "| 体系 | 检索情况 | 当前保留的目标应用线索 | 全文核验 |",
        "|---|---|---|---|",
    ]
    for formula in candidates:
        successful = any(
            formula in status.submitted_materials
            and formula not in status.failed_materials
            for status in result.provider_statuses
        )
        failed = any(
            formula in status.failed_materials for status in result.provider_statuses
        )
        state = (
            "已检索（有限结果）" if successful else "查询失败" if failed else "未检索"
        )
        if result.retrieval_mode == "cache" and successful:
            state = "历史已检索（本次复用）"
        leads = sum(
            p.relevance_level in {"core", "high"}
            and _paper_mentions_formula(p, formula)
            for p in result.papers
        )
        note = f"{leads} 篇题名/摘要线索" if leads else "当前保留结果暂无目标应用线索"
        review = "待上传全文核验" if leads else "尚无目标应用全文候选"
        lines.append(f"| {formula} | {state} | {note} | {review} |")
    lines.extend(
        (
            "",
            "线索数量只统计本次保留结果中与体系和目标应用相关的题名/摘要，"
            "不代表完整召回；暂无线索不代表没有相关论文。检索成功不等于全文核验完成，"
            "掺杂样品、物相与实验指标仍需全文确认。",
            "同一论文可能涉及多个体系，各体系线索数不能相加作为总篇数。",
        )
    )
    return "\n".join(lines)
