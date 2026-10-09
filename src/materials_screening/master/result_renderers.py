"""Pluggable presentation registry for unified multi-agent results."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from materials_screening.master.application_contracts import UnifiedResultEnvelope
from materials_screening.master.cross_agent import CrossAgentResult
from materials_screening.master.data_analysis_handoffs import (
    SourcePartitionedSynthesis,
    render_source_partitioned_synthesis,
)

ResultRenderer = Callable[[UnifiedResultEnvelope], str]


class ResultRendererRegistry:
    """Static renderer registry keyed by a validated ``result_type``."""

    def __init__(
        self,
        renderers: Sequence[tuple[str, ResultRenderer]] = (),
    ) -> None:
        self._renderers: dict[str, ResultRenderer] = {}
        for result_type, renderer in renderers:
            self.register(result_type, renderer)

    def register(self, result_type: str, renderer: ResultRenderer) -> None:
        if result_type in self._renderers:
            raise ValueError(f"duplicate result renderer: {result_type!r}")
        self._renderers[result_type] = renderer

    def render(self, envelope: UnifiedResultEnvelope) -> str:
        try:
            renderer = self._renderers[envelope.result_type]
        except KeyError as exc:
            raise KeyError(
                f"unknown result renderer: {envelope.result_type!r}"
            ) from exc
        return renderer(envelope)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._renderers))


def render_markdown_result(envelope: UnifiedResultEnvelope) -> str:
    """Render the common ``result.markdown`` application payload."""

    value = envelope.result.get("markdown")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("markdown result requires a non-blank result.markdown")
    return value


def render_cross_agent_result(envelope: UnifiedResultEnvelope) -> str:
    """Render the MA-4 typed literature/database result with source boundaries."""

    result = CrossAgentResult.model_validate(envelope.result)
    lines = [
        "## 论文线索与材料数据库联合结果",
        "",
        "论文中的组成仅作为检索线索，不代表该化合物已被 Materials Project 确认。",
        "",
        "### 论文材料线索",
        "",
    ]
    if not result.clues:
        lines.append("未提取到可安全转换的材料组成线索。")
    for index, clue in enumerate(result.clues, 1):
        review = "已审核" if clue.review_status == "approved" else "待审核"
        formulas = "、".join(clue.formula_candidates) or "未识别明确化学式"
        lines.extend(
            (
                f"{index}. **{clue.paper_title}**",
                f"   - 元素检索条件：{', '.join(clue.required_elements)}",
                f"   - 论文式样线索：{formulas}",
                f"   - 证据：第 {clue.evidence_page} 页（{review}）",
            )
        )
        for note in clue.inference_notes:
            lines.append(f"   - 转换说明：{note}")
    lines.extend(("", "### Materials Project 候选", ""))
    if result.candidates:
        lines.extend(
            (
                "| 论文线索 | Material ID | 化学式 | 稳定 | "
                "E hull (eV/atom) | 带隙 (eV) |",
                "|---|---|---|---:|---:|---:|",
            )
        )
        clue_numbers = {
            clue.clue_id: index for index, clue in enumerate(result.clues, 1)
        }
        for item in result.candidates:
            lines.append(
                "| {clue} | {mid} | {formula} | {stable} | {hull} | {gap} |".format(
                    clue=clue_numbers.get(item.clue_id, "?"),
                    mid=item.material_id,
                    formula=item.formula_pretty,
                    stable=_display(item.is_stable),
                    hull=_display(item.energy_above_hull_ev_atom),
                    gap=_display(item.band_gap_ev),
                )
            )
    else:
        lines.append("当前结构化条件未返回数据库候选。")
    if result.query_ids:
        lines.extend(("", "数据库查询快照：" + "、".join(result.query_ids)))
    if result.warnings:
        lines.extend(("", "### 证据边界", ""))
        lines.extend(f"- {warning}" for warning in result.warnings)
    return "\n".join(lines)


def render_cross_agent_analysis_result(envelope: UnifiedResultEnvelope) -> str:
    result = SourcePartitionedSynthesis.model_validate(envelope.result)
    return render_source_partitioned_synthesis(result)


def _display(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def default_renderer_registry() -> ResultRendererRegistry:
    """Return MA-1 renderers without importing any UI framework."""

    return ResultRendererRegistry(
        (
            ("materials_answer", render_markdown_result),
            ("literature_answer", render_markdown_result),
            ("analysis_answer", render_markdown_result),
            ("system_message", render_markdown_result),
            ("cross_agent_answer", render_cross_agent_result),
            ("cross_agent_analysis_answer", render_cross_agent_analysis_result),
        )
    )
