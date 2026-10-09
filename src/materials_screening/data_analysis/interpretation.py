"""Deterministic researcher-facing interpretation of statistical results."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from materials_screening.data_analysis.dataset_store import DatasetStore
from materials_screening.data_analysis.models import AnalysisResult
from materials_screening.data_analysis.scientific import (
    ScientificAnalysisBrief,
    ScientificBriefService,
)
from materials_screening.data_analysis.service import DataAnalysisService

EvidenceStrength = Literal["有限", "中等", "较强"]


class ScientificInterpretation(BaseModel):
    """Bounded narrative derived only from a brief, inspection, and result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    research_question: str
    direct_answer: str
    key_findings: tuple[str, ...] = Field(default=(), max_length=20)
    evidence_strength: EvidenceStrength
    evidence_explanation: str
    practical_significance: str
    assumption_summary: tuple[str, ...] = Field(default=(), max_length=20)
    limitations: tuple[str, ...] = Field(default=(), max_length=20)
    recommendations: tuple[str, ...] = Field(default=(), max_length=20)
    method_summary: str


class ScientificInterpretationService:
    """Explain deterministic results without inventing mechanisms or causality."""

    def __init__(self, store: DatasetStore) -> None:
        self.store = store
        self.inspection = DataAnalysisService(store)

    def interpret(
        self,
        brief: ScientificAnalysisBrief,
        result: AnalysisResult,
    ) -> ScientificInterpretation:
        ScientificBriefService.require_confirmed(brief)
        if brief.dataset_id != result.dataset_id:
            raise ValueError("brief and analysis result belong to different datasets")
        inspection = self.inspection.inspect_dataset(brief.dataset_id, limit=1)
        response = brief.response_variables[0]
        unit = brief.units.get(response, "")
        if result.analysis_type == "statistical_test":
            direct_answer = _statistical_answer(brief, result)
            findings = _statistical_findings(result, unit)
            assumptions = _assumption_summary(result)
        elif result.analysis_type == "regression":
            direct_answer = _regression_answer(brief, result)
            findings = _regression_findings(result, unit)
            assumptions = _regression_diagnostics(result)
        else:
            direct_answer = (
                "当前结果描述了数据特征，但尚未提供回答推断性研究问题的证据。"
            )
            findings = _descriptive_findings(result, unit)
            assumptions = ()
        strength, explanation = _evidence_strength(result)
        limitations = _limitations(brief, result, inspection)
        recommendations = _recommendations(brief, result, inspection)
        return ScientificInterpretation(
            research_question=brief.research_question,
            direct_answer=direct_answer,
            key_findings=findings,
            evidence_strength=strength,
            evidence_explanation=explanation,
            practical_significance=_practical_statement(brief, result),
            assumption_summary=assumptions,
            limitations=limitations,
            recommendations=recommendations,
            method_summary=_method_summary(result),
        )


def render_scientific_interpretation(value: ScientificInterpretation) -> str:
    """Render a plain-language narrative before technical statistical details."""

    lines = [
        "**科研结论摘要**",
        "",
        f"**研究问题：** {value.research_question}",
        "",
        f"**直接回答：{value.direct_answer}**",
        "",
        f"**证据强度：{value.evidence_strength}。**{value.evidence_explanation}",
        "",
        "**主要发现**",
        "",
    ]
    lines.extend(f"- {item}" for item in value.key_findings)
    lines.extend(("", "**实际科研意义**", "", value.practical_significance))
    if value.assumption_summary:
        lines.extend(("", "**方法前提与诊断**", ""))
        lines.extend(f"- {item}" for item in value.assumption_summary)
    lines.extend(("", "**不能由本次分析说明**", ""))
    lines.extend(f"- {item}" for item in value.limitations)
    lines.extend(("", "**建议的下一步**", ""))
    lines.extend(
        f"{index}. {item}"
        for index, item in enumerate(value.recommendations, 1)
    )
    lines.extend(("", "**方法说明**", "", value.method_summary))
    return "\n".join(lines)


def _statistical_answer(
    brief: ScientificAnalysisBrief, result: AnalysisResult
) -> str:
    significant = result.summary.get("significant") is True
    if significant:
        if brief.design == "independent_groups":
            return "当前数据支持不同组的该指标存在统计学差异。"
        if brief.design in {"paired", "repeated_measures"}:
            return "当前数据支持不同条件下的该指标存在统计学差异。"
        return "当前数据支持所比较对象之间存在统计学差异。"
    return (
        "当前数据没有提供足够证据确认差异；这不等同于证明各组或条件完全相同。"
    )


def _statistical_findings(
    result: AnalysisResult, unit: str
) -> tuple[str, ...]:
    findings: list[str] = []
    rows = result.summary.get("groups") or result.summary.get("conditions") or []
    if rows:
        ordered = sorted(
            (row for row in rows if row.get("mean") is not None),
            key=lambda row: float(row["mean"]),
            reverse=True,
        )
        if ordered:
            ranking = " > ".join(
                f"{row.get('label')}（{_number(row.get('mean'))}{_unit(unit)}）"
                for row in ordered
            )
            findings.append(f"按均值排序为：{ranking}。")
    p_value = result.summary.get("p_value")
    statistic = result.summary.get("statistic")
    degrees = result.summary.get("degrees_of_freedom")
    findings.append(
        f"总体检验统计量为 {_number(statistic)}，自由度为 {degrees}，"
        f"p={_number(p_value)}。"
    )
    effect = result.summary.get("effect_size") or {}
    if effect:
        findings.append(
            f"效应量为 {_effect_label(str(effect.get('name')))}="
            f"{_number(effect.get('value'))}；这是统计效应指标，"
            "不能替代领域内的实际意义标准。"
        )
    means = {
        str(row.get("label")): float(row["mean"])
        for row in rows
        if row.get("mean") is not None
    }
    for comparison in result.summary.get("post_hoc", [])[:8]:
        left = str(
            comparison.get("group_a", comparison.get("condition_a", "—"))
        )
        right = str(
            comparison.get("group_b", comparison.get("condition_b", "—"))
        )
        adjusted = comparison.get("adjusted_p_value")
        significant = comparison.get("significant") is True
        difference_text = ""
        if left in means and right in means:
            difference = means[left] - means[right]
            higher, lower = (left, right) if difference >= 0 else (right, left)
            interval = comparison.get("difference_confidence_interval")
            if interval and difference < 0:
                interval = [-float(interval[1]), -float(interval[0])]
            elif interval:
                interval = [float(interval[0]), float(interval[1])]
            difference_text = (
                f"{higher} 比 {lower} 高 {abs(difference):.4g}{_unit(unit)}"
                + (
                    f"，差值95%区间为 [{_number(interval[0])}, "
                    f"{_number(interval[1])}]{_unit(unit)}"
                    if interval
                    else ""
                )
            )
        verdict = "达到统计学显著" if significant else "未达到统计学显著"
        findings.append(
            f"{left} 与 {right}：{difference_text or '完成两两比较'}；"
            f"Holm校正后 p={_number(adjusted)}，{verdict}。"
        )
    dropped = result.summary.get("dropped_incomplete_subjects")
    if dropped:
        findings.append(f"有 {dropped} 个样本因条件记录不完整而被排除。")
    return tuple(findings)


def _regression_answer(
    brief: ScientificAnalysisBrief, result: AnalysisResult
) -> str:
    if result.summary.get("significant") is True:
        return "当前数据支持响应指标与连续预测因素之间存在线性关联。"
    return "当前数据没有提供足够证据确认两者存在线性关联。"


def _regression_findings(
    result: AnalysisResult, unit: str
) -> tuple[str, ...]:
    interval = result.summary.get("slope_confidence_interval") or [None, None]
    return (
        f"预测因素每增加1单位，响应指标平均变化 {_number(result.summary.get('slope'))}"
        f"{_unit(unit)}；斜率95%区间为 [{_number(interval[0])}, "
        f"{_number(interval[1])}]{_unit(unit)}。",
        f"模型R²={_number(result.summary.get('r_squared'))}，"
        f"p={_number(result.summary.get('p_value'))}，"
        f"有效样本数为 {result.summary.get('n', '—')}。",
    )


def _descriptive_findings(
    result: AnalysisResult, unit: str
) -> tuple[str, ...]:
    rows = result.summary.get("statistics") or []
    if not rows:
        return ("当前结果未包含可解释的描述统计。",)
    row = rows[0]
    return (
        f"有效样本数为 {row.get('count', '—')}，均值为 "
        f"{_number(row.get('mean'))}{_unit(unit)}，标准差为 "
        f"{_number(row.get('std'))}{_unit(unit)}。",
    )


def _evidence_strength(result: AnalysisResult) -> tuple[EvidenceStrength, str]:
    if result.analysis_type not in {"statistical_test", "regression"}:
        return "有限", "当前结果以描述为主，不能单独支持推断性结论。"
    significant = result.summary.get("significant") is True
    p_value = float(result.summary.get("p_value", 1.0))
    if not significant:
        return (
            "有限",
            "未达到预设显著性阈值，因此对“存在差异或关联”的支持有限；"
            "结果也不能证明不存在差异。",
        )
    score = 1 if p_value < 0.05 else 0
    if p_value < 0.01:
        score += 1
    if p_value < 0.001:
        score += 1
    effect = result.summary.get("effect_size") or result.summary.get(
        "standardized_effect"
    ) or {}
    magnitude = _effect_magnitude(str(effect.get("name")), effect.get("value"))
    if magnitude is not None:
        score += {"small": 0, "medium": 1, "large": 2}[magnitude]
    sample_count = _result_sample_count(result)
    if sample_count >= 30:
        score += 1
    if result.warnings:
        score -= 2
    if score >= 5:
        return (
            "较强",
            "p值较小、效应量较大，且当前结果未记录明显方法警告；"
            "该等级只评价当前数据中的统计证据，不代表因果证据。",
        )
    if score >= 2:
        return (
            "中等",
            "结果达到统计学显著，但样本量、效应精度或方法前提仍限制结论强度。",
        )
    return "有限", "虽达到统计学显著，但当前证据仍受样本或方法警告限制。"


def _assumption_summary(result: AnalysisResult) -> tuple[str, ...]:
    assumptions = result.summary.get("assumptions") or {}
    messages: list[str] = []
    normality = assumptions.get("normality") or []
    if normality:
        failed = [
            str(item.get("group"))
            for item in normality
            if item.get("passed") is False
        ]
        unknown = [
            str(item.get("group"))
            for item in normality
            if item.get("passed") is None
        ]
        if failed:
            messages.append(f"以下组未通过正态性检查：{'、'.join(failed)}。")
        elif unknown:
            messages.append(
                f"以下组样本不足，无法可靠检查正态性：{'、'.join(unknown)}。"
            )
        else:
            messages.append(
                "各组未发现明显正态性违背；小样本检验能力有限，不能据此证明完全正态。"
            )
    variance = assumptions.get("equal_variance") or {}
    if variance.get("passed") is True:
        messages.append(
            f"方差齐性检查未发现明显违背（p={_number(variance.get('p_value'))}）。"
        )
    elif variance.get("passed") is False:
        messages.append(
            f"方差齐性检查未通过（p={_number(variance.get('p_value'))}）。"
        )
    difference_normality = assumptions.get("normality_of_differences") or {}
    if difference_normality:
        passed = difference_normality.get("passed")
        messages.append(
            "配对差值未发现明显正态性违背。"
            if passed is True
            else "配对差值的正态性前提未得到充分支持。"
        )
    return tuple(messages)


def _regression_diagnostics(result: AnalysisResult) -> tuple[str, ...]:
    diagnostics = result.summary.get("diagnostics") or {}
    messages: list[str] = []
    normality = diagnostics.get("residual_normality") or {}
    if normality:
        messages.append(
            "残差未发现明显正态性违背。"
            if normality.get("passed") is True
            else "残差正态性前提未得到充分支持。"
        )
    variance = diagnostics.get("absolute_residual_fitted_spearman") or {}
    if variance.get("p_value") is not None:
        messages.append(
            "残差大小与拟合值的关联检查 p="
            f"{_number(variance.get('p_value'))}；p<0.05时应警惕异方差。"
        )
    outliers = diagnostics.get("large_standardized_residual_count", 0)
    messages.append(f"绝对标准化残差大于3的观测数为 {outliers}。")
    return tuple(messages)


def _practical_statement(
    brief: ScientificAnalysisBrief, result: AnalysisResult
) -> str:
    response = brief.response_variables[0]
    threshold = brief.practical_thresholds.get(response)
    unit = brief.units.get(response, "")
    if threshold is None:
        return (
            "尚未设置最小实际意义阈值。因此，即使统计学显著，"
            "也不能自动断言差异已经达到科研、工程或应用上的重要程度。"
        )
    observed = _representative_change(result)
    if observed is None:
        return (
            f"预设阈值为 {threshold:g}{_unit(unit)}，但当前结果不适合自动计算"
            "可与该阈值直接比较的变化量。"
        )
    verdict = "达到" if observed >= threshold else "未达到"
    return (
        f"预设阈值为 {threshold:g}{_unit(unit)}；观察到的代表性变化为 "
        f"{observed:.4g}{_unit(unit)}，{verdict}该阈值。"
    )


def _limitations(
    brief: ScientificAnalysisBrief, result: AnalysisResult, inspection: Any
) -> tuple[str, ...]:
    limitations: list[str] = []
    if brief.design == "independent_groups":
        limitations.append(
            "数据表本身不能验证随机分组、样品独立性或制备顺序；"
            "因此这里报告组间差异，不直接宣称处理造成了变化。"
        )
    elif brief.design in {"paired", "repeated_measures"}:
        limitations.append(
            "结果依赖样本ID配对正确，并可能受到测量顺序、时间趋势或残留效应影响。"
        )
    elif brief.design == "continuous_relationship":
        limitations.append("线性回归描述关联，不能单独证明预测因素导致响应变化。")
    if _result_sample_count(result) < 30:
        limitations.append("有效样本量少于30，效应估计和前提检查可能不稳定。")
    if inspection.issues:
        limitations.append(
            "数据存在质量提示；缺失、重复或异常候选需结合原始实验记录复核。"
        )
    response = set(brief.response_variables)
    assigned = response | set(brief.covariates)
    if brief.group_variable:
        assigned.add(brief.group_variable)
    if brief.subject_id_variable:
        assigned.add(brief.subject_id_variable)
    other_numeric = [
        column.name
        for column in inspection.columns
        if column.inferred_type == "numeric"
        and column.name not in assigned
        and column.unique_count > 1
    ][:3]
    if other_numeric and brief.design == "independent_groups":
        limitations.append(
            "当前单因素分析没有控制其他数值字段（如"
            f"{'、'.join(other_numeric)}）；它们是否构成混杂需由实验设计判断。"
        )
    if result.warnings:
        limitations.append("统计过程记录了方法警告，结论需结合稳健方法复核。")
    return tuple(limitations)


def _recommendations(
    brief: ScientificAnalysisBrief, result: AnalysisResult, inspection: Any
) -> tuple[str, ...]:
    recommendations = ["回查实验记录，确认单位、分组、样品独立性和缺失值含义。"]
    if not brief.practical_thresholds:
        recommendations.append(
            "由课题负责人预先给出最小有意义差异，再判断统计结果是否具有科研价值。"
        )
    if inspection.issues:
        recommendations.append("逐条复核质量提示，并保留清洗前后的可追溯数据版本。")
    if brief.design == "independent_groups":
        recommendations.append(
            "绘制按组分布图并检查异常观测；若其他工艺字段在组间不平衡，"
            "使用预先指定的多因素模型复核。"
        )
    elif brief.design in {"paired", "repeated_measures"}:
        recommendations.append("核对每个样本的条件顺序和缺失配对，并报告被排除样本。")
    elif brief.design == "continuous_relationship":
        recommendations.append("检查散点图、非线性趋势和高影响观测，再决定是否扩展模型。")
    recommendations.append(
        "在独立批次或新增样品中复现主要结果后，再讨论材料机理或推广结论。"
    )
    return tuple(recommendations)


def _method_summary(result: AnalysisResult) -> str:
    labels = {
        "welch_t": "Welch t检验",
        "student_t": "独立样本t检验",
        "mann_whitney": "Mann–Whitney U检验",
        "anova": "单因素方差分析",
        "kruskal_wallis": "Kruskal–Wallis检验",
        "paired_t": "配对t检验",
        "wilcoxon": "Wilcoxon符号秩检验",
        "friedman": "Friedman重复测量检验",
        "simple_linear_regression": "简单线性回归",
        "describe": "描述统计",
    }
    correction = result.summary.get("multiple_testing_correction")
    suffix = "；两两比较使用Holm校正控制多重比较。" if correction == "holm" else "。"
    return (
        f"本次使用{labels.get(result.method, result.method)}，"
        f"显著性阈值为 {result.summary.get('alpha', '—')}{suffix}"
    )


def _effect_magnitude(name: str, value: Any) -> str | None:
    if value is None:
        return None
    absolute = abs(float(value))
    if name in {"cohens_d", "cohens_dz"}:
        return "large" if absolute >= 0.8 else "medium" if absolute >= 0.5 else "small"
    if name in {
        "rank_biserial_correlation",
        "matched_rank_biserial_correlation",
        "kendalls_w",
        "standardized_beta",
    }:
        return "large" if absolute >= 0.5 else "medium" if absolute >= 0.3 else "small"
    if name in {"eta_squared", "epsilon_squared"}:
        if absolute >= 0.14:
            return "large"
        return "medium" if absolute >= 0.06 else "small"
    return None


def _result_sample_count(result: AnalysisResult) -> int:
    if result.summary.get("pairs") is not None:
        return int(result.summary["pairs"])
    rows = result.summary.get("groups") or result.summary.get("conditions") or []
    if rows:
        return sum(int(row.get("n", 0)) for row in rows)
    if result.summary.get("n") is not None:
        return int(result.summary["n"])
    statistics = result.summary.get("statistics") or []
    return max((int(row.get("count", 0)) for row in statistics), default=0)


def _representative_change(result: AnalysisResult) -> float | None:
    for key in ("mean_difference", "median_difference", "slope"):
        if result.summary.get(key) is not None:
            return abs(float(result.summary[key]))
    rows = result.summary.get("groups") or result.summary.get("conditions") or []
    means = [float(row["mean"]) for row in rows if row.get("mean") is not None]
    return max(means) - min(means) if means else None


def _effect_label(name: str) -> str:
    return {
        "eta_squared": "η²",
        "epsilon_squared": "ε²",
        "cohens_d": "Cohen's d",
        "cohens_dz": "Cohen's dz",
        "rank_biserial_correlation": "秩二列相关",
        "matched_rank_biserial_correlation": "配对秩二列相关",
        "kendalls_w": "Kendall's W",
        "standardized_beta": "标准化β",
    }.get(name, name or "效应量")


def _number(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _unit(value: str) -> str:
    return f" {value}" if value else ""
