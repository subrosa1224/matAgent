"""Render descriptive numbers from validated tool rows, never model prose."""

from __future__ import annotations

import json
import math
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from materials_screening.agent.models import AgentFinalDraft, AgentFinalStatus
from materials_screening.data_analysis.models import (
    AnalysisResult,
    DatasetReference,
    JsonScalar,
)


class _StatisticRow(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    column: str = Field(min_length=1, max_length=256)
    group: JsonScalar
    count: int = Field(ge=0, le=100_000)
    missing_count: int = Field(ge=0, le=100_000)
    mean: float | None
    std: float | None
    min: float | None
    max: float | None
    quantiles: dict[str, float | None]

    @model_validator(mode="after")
    def validate_numbers(self) -> Self:
        values = [self.mean, self.std, self.min, self.max, *self.quantiles.values()]
        if any(value is not None and not math.isfinite(value) for value in values):
            raise ValueError("statistics must be finite or null")
        if isinstance(self.group, float) and not math.isfinite(self.group):
            raise ValueError("group label must be finite")
        if self.std is not None and self.std < 0:
            raise ValueError("standard deviation must not be negative")
        if self.count == 0 and any(value is not None for value in values):
            raise ValueError("empty samples must not have numerical statistics")
        if self.count > 0 and any(
            value is None
            for value in (self.mean, self.min, self.max, *self.quantiles.values())
        ):
            raise ValueError("non-empty samples require numerical statistics")
        return self


def render_descriptive_report(
    analysis: AnalysisResult,
    dataset: DatasetReference,
) -> AgentFinalDraft:
    """Check row coverage and build a bounded, source-linked Chinese report."""

    if (
        analysis.dataset_id != dataset.dataset_id
        or analysis.analysis_type != "descriptive"
        or analysis.method != "describe"
    ):
        raise ValueError("分析结果与数据集或描述统计方法不匹配")
    columns = analysis.parameters.get("columns")
    if (
        not isinstance(columns, list)
        or not 1 <= len(columns) <= 50
        or any(not isinstance(value, str) for value in columns)
        or len(set(columns)) != len(columns)
    ):
        raise ValueError("描述统计字段参数无效")
    raw_rows = analysis.summary.get("statistics")
    if not isinstance(raw_rows, list) or len(raw_rows) > 1000:
        raise ValueError("统计行缺失或超出上限")
    try:
        rows = [_StatisticRow.model_validate(row) for row in raw_rows]
    except ValueError as exc:
        raise ValueError("统计行格式、计数或数值无效") from exc
    group_by = analysis.parameters.get("group_by")
    quantiles = analysis.parameters.get("quantiles")
    if not isinstance(quantiles, list) or not quantiles:
        raise ValueError("分位数参数缺失")
    quantile_keys = [str(float(value)) for value in quantiles]
    groups: dict[str, tuple[JsonScalar, int, set[str]]] = {}
    for row in rows:
        if row.column not in columns or set(row.quantiles) != set(quantile_keys):
            raise ValueError("统计字段或分位数与分析参数不一致")
        if group_by is None and row.group is not None:
            raise ValueError("未分组统计不应含分组标签")
        key = json.dumps(row.group, ensure_ascii=False)
        size = row.count + row.missing_count
        if key not in groups:
            groups[key] = (row.group, size, set())
        _, expected_size, seen_columns = groups[key]
        if row.column in seen_columns or size != expected_size:
            raise ValueError("重复统计行或同组记录数不一致")
        seen_columns.add(row.column)
    if any(seen != set(columns) for _, _, seen in groups.values()):
        raise ValueError("分组缺少目标字段的统计行")
    if sum(size for _, size, _ in groups.values()) != dataset.row_count:
        raise ValueError("分组样本数与数据集总行数不一致")
    if not rows and dataset.row_count != 0:
        raise ValueError("非空数据集没有统计结果")

    lines = [
        f"数据集：`{dataset.dataset_id}`；分析：`{analysis.analysis_id}`。",
        f"样本数：{dataset.row_count}；"
        + (
            f"按 `{_cell(group_by)}` 分组，分组数：{len(groups)}。"
            if group_by is not None
            else "未分组，按整个数据集统计。"
        ),
        "计数已校验：各组记录数合计等于数据集行数；"
        "记录数=有效数+缺失/非有限数，不把不同字段的样本数重复相加。",
        "描述统计仅反映当前数据分布，不推断器件性能或材料应用可行性。",
        "\n| 分组 | 记录数 |\n|---|---:|",
    ]
    shown_groups = 0
    for label, size, _ in groups.values():
        if sum(map(len, lines)) > 2500:
            break
        lines.append(f"| {_cell(label) if group_by is not None else '整体'} | {size} |")
        shown_groups += 1
    if shown_groups != len(groups):
        lines.append(
            f"仅预览 {shown_groups}/{len(groups)} 组；总数校验使用全部统计行。"
        )
    lines.append("\n各字段有效数与缺失/非有限数：")
    shown_fields = 0
    for column in columns:
        if sum(map(len, lines)) > 4000:
            break
        selected = [row for row in rows if row.column == column]
        lines.append(
            f"{_cell(column)}：有效={sum(row.count for row in selected)}，"
            f"缺失/非有限={sum(row.missing_count for row in selected)}。"
        )
        shown_fields += 1
    if shown_fields != len(columns):
        lines.append(
            f"字段计数仅展示 {shown_fields}/{len(columns)} 项；校验使用全部字段。"
        )
    lines.append(
        "\n| 字段 | 分组 | 有效数 | 缺失/非有限 | 均值 | 标准差 | 最小值 | "
        + " | ".join(f"Q{float(value):g}" for value in quantiles)
        + " | 最大值 |"
    )
    lines.append("|" + "---|" * (8 + len(quantiles)))
    shown_rows = 0
    for row in rows:
        numbers = [row.mean, row.std, row.min]
        numbers.extend(row.quantiles[key] for key in quantile_keys)
        numbers.append(row.max)
        line = (
            f"| {_cell(row.column)} | "
            f"{_cell(row.group) if group_by is not None else '整体'} | {row.count} | "
            f"{row.missing_count} | "
            + " | ".join(_number(value) for value in numbers)
            + " |"
        )
        if sum(map(len, lines)) + len(line) > 7000:
            break
        lines.append(line)
        shown_rows += 1
    warnings = list(analysis.warnings)
    if shown_rows != len(rows):
        note = f"统计表仅展示 {shown_rows}/{len(rows)} 行；完整结果保留在上述分析 ID。"
        lines.append(note)
        warnings.append(note)
    lines.append("数值以10位有效数字展示；— 表示无可用统计量，不表示0。")
    return AgentFinalDraft(
        status=AgentFinalStatus.COMPLETED,
        answer="\n".join(lines),
        evidence_ids=[analysis.evidence_id],
        warnings=warnings,
    )


def _number(value: float | None) -> str:
    return "—" if value is None else format(value, ".10g")


def _cell(value: JsonScalar) -> str:
    if value is None:
        return "缺失分组"
    text = str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")
    return text if len(text) <= 80 else text[:80] + "…（标签截短）"
