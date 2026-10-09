"""One-click, read-only exploratory analysis for newly registered datasets."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from materials_screening.data_analysis.models import (
    AnalysisResult,
    DatasetInspection,
)
from materials_screening.data_analysis.service import DataAnalysisService
from materials_screening.data_analysis.statistics import DataStatisticsService


class ExploratoryAnalysisBundle(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    inspection: DatasetInspection
    descriptive: AnalysisResult | None = None
    correlation: AnalysisResult | None = None
    recommendations: tuple[str, ...] = ()

    @property
    def analysis_ids(self) -> tuple[str, ...]:
        return tuple(
            result.analysis_id
            for result in (self.descriptive, self.correlation)
            if result is not None
        )


class DataExplorationService:
    """Run safe inspection, description, and optional correlation only."""

    def __init__(
        self,
        inspection: DataAnalysisService,
        statistics: DataStatisticsService,
    ) -> None:
        self._inspection = inspection
        self._statistics = statistics

    def explore(self, dataset_id: str) -> ExploratoryAnalysisBundle:
        inspection = self._inspection.inspect_dataset(dataset_id, limit=20)
        numeric = tuple(
            column.name
            for column in inspection.columns
            if column.inferred_type == "numeric" and column.unique_count > 1
        )
        descriptive = (
            self._statistics.describe_dataset(dataset_id, columns=numeric[:50])
            if numeric
            else None
        )
        correlation = (
            self._statistics.analyze_correlations(
                dataset_id,
                columns=numeric[:20],
                method="pearson",
            )
            if len(numeric) >= 2
            else None
        )
        recommendations: list[str] = []
        issue_codes = {issue.code for issue in inspection.issues}
        if "missing_values" in issue_codes:
            recommendations.append("确认缺失值含义后，再显式选择删除或填补策略。")
        if "duplicate_rows" in issue_codes:
            recommendations.append("确认重复记录是否为重复实验，再决定是否去重。")
        if "iqr_outlier_candidates" in issue_codes:
            recommendations.append("先核查异常值来源，不要直接自动删除。")
        categorical = [
            column.name
            for column in inspection.columns
            if column.inferred_type in {"categorical", "string"}
            and 1 < column.unique_count <= 20
        ]
        if categorical and numeric:
            recommendations.append(
                f"可按 {categorical[0]} 分组比较 {numeric[0]}，"
                "显著性检验需由用户明确选择。"
            )
        if len(numeric) >= 2:
            recommendations.append("可选择两个数值字段生成散点图或相关热力图。")
        if not recommendations:
            recommendations.append("数据结构可用，可继续选择字段执行统计或绘图。")
        return ExploratoryAnalysisBundle(
            inspection=inspection,
            descriptive=descriptive,
            correlation=correlation,
            recommendations=tuple(recommendations),
        )


def render_exploration_markdown(bundle: ExploratoryAnalysisBundle) -> str:
    dataset = bundle.inspection.dataset
    lines = [
        "## 一键探索分析",
        "",
        f"数据集：`{dataset.dataset_id}` · {dataset.row_count} 行 × "
        f"{dataset.column_count} 列",
        "",
        "### 数据质量",
        "",
    ]
    if bundle.inspection.issues:
        lines.extend(f"- {issue.message}" for issue in bundle.inspection.issues[:20])
    else:
        lines.append("- 未发现缺失、重复、常量列或 IQR 异常值提示。")
    if bundle.descriptive is not None:
        statistics = bundle.descriptive.summary.get("statistics", [])
        lines.extend(
            (
                "",
                "### 数值摘要",
                "",
                "| 字段 | n | 均值 | 中位数 | 标准差 |",
                "|---|---:|---:|---:|---:|",
            )
        )
        for row in statistics[:20]:
            lines.append(
                "| {column} | {count} | {mean} | {median} | {std} |".format(
                    column=row.get("column", "—"),
                    count=row.get("count", "—"),
                    mean=_display(row.get("mean")),
                    median=_display(row.get("median")),
                    std=_display(row.get("std")),
                )
            )
    if bundle.correlation is not None:
        pairs = [
            row
            for row in bundle.correlation.summary.get("pairs", [])
            if row.get("x") != row.get("y")
            and row.get("coefficient") is not None
        ]
        pairs.sort(key=lambda row: abs(float(row["coefficient"])), reverse=True)
        lines.extend(("", "### 最强相关关系", ""))
        if pairs:
            for row in pairs[:5]:
                lines.append(
                    f"- {row['x']} ↔ {row['y']}："
                    f"r={_display(row['coefficient'])}，n={row['n']}"
                )
            lines.append("- 相关性不代表因果关系。")
        else:
            lines.append("- 当前有效样本不足以形成字段间相关系数。")
    lines.extend(("", "### 推荐下一步", ""))
    lines.extend(f"- {item}" for item in bundle.recommendations)
    return "\n".join(lines)


def _display(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)
