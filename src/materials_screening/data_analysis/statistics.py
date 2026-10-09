"""Deterministic descriptive, correlation and inferential statistics."""

from __future__ import annotations

import math
import uuid
from collections.abc import Sequence
from typing import Any, Literal

import pandas as pd  # type: ignore[import-untyped]
from scipy import stats  # type: ignore[import-untyped]

from materials_screening.data_analysis.dataset_store import DatasetStore
from materials_screening.data_analysis.models import AnalysisResult, JsonScalar

CorrelationMethod = Literal["pearson", "spearman"]
StatisticalTestMethod = Literal[
    "student_t",
    "welch_t",
    "paired_t",
    "wilcoxon",
    "mann_whitney",
    "anova",
    "kruskal_wallis",
    "friedman",
]


class DataStatisticsService:
    """Execute source-linked statistics and persist strict JSON results."""

    def __init__(self, store: DatasetStore) -> None:
        self.store = store

    def describe_dataset(
        self,
        dataset_id: str,
        *,
        columns: Sequence[str],
        group_by: str | None = None,
        quantiles: Sequence[float] = (0.25, 0.5, 0.75),
        evidence_id: str | None = None,
    ) -> AnalysisResult:
        frame = self.store.load_dataframe(dataset_id)
        selected = _validate_numeric_columns(frame, columns)
        checked_quantiles = _validate_quantiles(quantiles)
        if group_by is not None:
            _require_column(frame, group_by)
            group_count = int(frame[group_by].nunique(dropna=False))
            if group_count > 100:
                raise ValueError("group_by contains more than 100 groups")
            if group_count * len(selected) > 1000:
                raise ValueError("grouped description exceeds 1000 result rows")

        statistics: list[dict[str, Any]] = []
        if group_by is None:
            for column in selected:
                statistics.append(
                    _describe_series(frame[column], column, None, checked_quantiles)
                )
        else:
            grouped = frame.groupby(group_by, dropna=False, sort=True)
            for group_value, group_frame in grouped:
                safe_group = _json_scalar(group_value)
                for column in selected:
                    statistics.append(
                        _describe_series(
                            group_frame[column], column, safe_group, checked_quantiles
                        )
                    )
        return self._save_result(
            dataset_id=dataset_id,
            analysis_type="descriptive",
            method="describe",
            parameters={
                "columns": list(selected),
                "group_by": group_by,
                "quantiles": list(checked_quantiles),
                "missing_strategy": "drop_per_column",
            },
            summary={"statistics": statistics},
            evidence_id=evidence_id,
        )

    def analyze_correlations(
        self,
        dataset_id: str,
        *,
        columns: Sequence[str],
        method: CorrelationMethod = "pearson",
        evidence_id: str | None = None,
    ) -> AnalysisResult:
        frame = self.store.load_dataframe(dataset_id)
        selected = _validate_numeric_columns(
            frame, columns, minimum=2, maximum=40
        )
        if method not in {"pearson", "spearman"}:
            raise ValueError("correlation method must be pearson or spearman")
        warnings: list[str] = []
        pairs: list[dict[str, Any]] = []
        for left_index, left in enumerate(selected):
            for right in selected[left_index:]:
                coefficient: float | None
                if left == right:
                    values = _finite_numeric(frame[left])
                    count = len(values.index)
                    coefficient = 1.0 if count >= 2 and values.nunique() > 1 else None
                else:
                    pair = frame[[left, right]].copy()
                    pair[left] = pd.to_numeric(pair[left], errors="coerce")
                    pair[right] = pd.to_numeric(pair[right], errors="coerce")
                    pair = pair[
                        pair[left].map(lambda value: _is_finite_number(value))
                        & pair[right].map(lambda value: _is_finite_number(value))
                    ]
                    count = len(pair.index)
                    coefficient = None
                    if (
                        count >= 2
                        and pair[left].nunique() > 1
                        and pair[right].nunique() > 1
                    ):
                        raw = pair[left].corr(pair[right], method=method)
                        coefficient = _finite_or_none(raw)
                if count < 2 or coefficient is None:
                    warnings.append(
                        f"{left} 与 {right} 的有效样本不足或包含常量字段。"
                    )
                pairs.append(
                    {
                        "x": left,
                        "y": right,
                        "n": count,
                        "coefficient": coefficient,
                    }
                )
        return self._save_result(
            dataset_id=dataset_id,
            analysis_type="correlation",
            method=method,
            parameters={
                "columns": list(selected),
                "missing_strategy": "pairwise_complete",
            },
            summary={"pairs": pairs},
            warnings=_bounded_warnings(warnings),
            evidence_id=evidence_id,
        )

    def run_statistical_test(
        self,
        dataset_id: str,
        *,
        method: StatisticalTestMethod,
        response_column: str | None = None,
        group_column: str | None = None,
        groups: Sequence[JsonScalar] | None = None,
        paired_columns: tuple[str, str] | None = None,
        subject_id_column: str | None = None,
        condition_column: str | None = None,
        alpha: float = 0.05,
        evidence_id: str | None = None,
    ) -> AnalysisResult:
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1")
        frame = self.store.load_dataframe(dataset_id)
        if method in {"paired_t", "wilcoxon"}:
            if paired_columns is not None:
                left, right = _paired_samples(frame, paired_columns)
                pair_labels = paired_columns
                dropped_subjects = 0
            else:
                if not response_column or not subject_id_column or not condition_column:
                    raise ValueError(
                        f"{method} requires response, subject ID and condition columns"
                    )
                samples, condition_labels, dropped_subjects = _repeated_samples(
                    frame,
                    response_column=response_column,
                    subject_id_column=subject_id_column,
                    condition_column=condition_column,
                    required_conditions=2,
                )
                left, right = samples
                pair_labels = (
                    str(condition_labels[0]),
                    str(condition_labels[1]),
                )
            if method == "paired_t":
                summary, warnings = _paired_t(left, right, alpha, pair_labels)
            else:
                summary, warnings = _wilcoxon(left, right, alpha, pair_labels)
            summary["dropped_incomplete_subjects"] = dropped_subjects
            parameters: dict[str, Any] = {
                "paired_columns": list(paired_columns) if paired_columns else None,
                "response_column": response_column,
                "subject_id_column": subject_id_column,
                "condition_column": condition_column,
                "alpha": alpha,
                "missing_strategy": "drop_incomplete_pairs",
            }
        elif method == "friedman":
            if not response_column or not subject_id_column or not condition_column:
                raise ValueError(
                    "friedman requires response, subject ID and condition columns"
                )
            samples, repeated_labels, dropped_subjects = _repeated_samples(
                frame,
                response_column=response_column,
                subject_id_column=subject_id_column,
                condition_column=condition_column,
                minimum_conditions=3,
            )
            summary, warnings = _friedman(samples, repeated_labels, alpha)
            summary["dropped_incomplete_subjects"] = dropped_subjects
            parameters = {
                "response_column": response_column,
                "subject_id_column": subject_id_column,
                "condition_column": condition_column,
                "alpha": alpha,
                "missing_strategy": "complete_subjects_only",
            }
        else:
            if response_column is None or group_column is None:
                raise ValueError(
                    f"{method} requires response_column and group_column"
                )
            samples, group_labels = _group_samples(
                frame, response_column, group_column, groups
            )
            if method in {"student_t", "welch_t", "mann_whitney"}:
                if len(samples) != 2:
                    raise ValueError(f"{method} requires exactly two groups")
                summary, warnings = _two_group_test(
                    method, samples[0], samples[1], group_labels, alpha
                )
            elif method == "anova":
                summary, warnings = _anova(samples, group_labels, alpha)
            elif method == "kruskal_wallis":
                summary, warnings = _kruskal(samples, group_labels, alpha)
            else:
                raise ValueError(f"unsupported statistical test: {method}")
            parameters = {
                "response_column": response_column,
                "group_column": group_column,
                "groups": group_labels,
                "alpha": alpha,
                "missing_strategy": "drop_per_group",
            }
        return self._save_result(
            dataset_id=dataset_id,
            analysis_type="statistical_test",
            method=method,
            parameters=parameters,
            summary=summary,
            warnings=_bounded_warnings(warnings),
            evidence_id=evidence_id,
        )

    def run_linear_regression(
        self,
        dataset_id: str,
        *,
        response_column: str,
        predictor_column: str,
        alpha: float = 0.05,
        evidence_id: str | None = None,
    ) -> AnalysisResult:
        """Fit a bounded simple linear regression with residual diagnostics."""

        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1")
        frame = self.store.load_dataframe(dataset_id)
        _validate_numeric_columns(
            frame, (response_column, predictor_column), minimum=2
        )
        if response_column == predictor_column:
            raise ValueError("response and predictor columns must be different")
        pair = frame[[predictor_column, response_column]].copy()
        for column in pair.columns:
            pair[column] = pd.to_numeric(pair[column], errors="coerce")
        pair = pair[
            pair[predictor_column].map(_is_finite_number)
            & pair[response_column].map(_is_finite_number)
        ].astype(float)
        if len(pair.index) < 3:
            raise ValueError("linear regression requires at least 3 complete rows")
        if pair[predictor_column].nunique() < 2:
            raise ValueError("predictor column must not be constant")
        fitted = stats.linregress(pair[predictor_column], pair[response_column])
        slope = _finite_float(fitted.slope)
        intercept = _finite_float(fitted.intercept)
        predictions = intercept + slope * pair[predictor_column]
        residuals = pair[response_column] - predictions
        degrees = len(pair.index) - 2
        critical = float(stats.t.ppf(1 - alpha / 2, degrees))
        slope_interval = [
            slope - critical * _finite_float(fitted.stderr),
            slope + critical * _finite_float(fitted.stderr),
        ]
        normality = _normality_check(residuals, alpha)
        heteroscedasticity = stats.spearmanr(
            predictions, residuals.abs(), nan_policy="omit"
        )
        variance_p = _finite_or_none(heteroscedasticity.pvalue)
        residual_std = float(residuals.std(ddof=1))
        outlier_count = (
            int((residuals.abs() / residual_std > 3).sum())
            if residual_std > 0
            else 0
        )
        warnings: list[str] = []
        if normality["passed"] is False:
            warnings.append("回归残差未通过正态性检查。")
        if variance_p is not None and variance_p < alpha:
            warnings.append("残差大小随拟合值变化，可能存在异方差。")
        if outlier_count:
            warnings.append(f"检测到 {outlier_count} 个绝对标准化残差大于 3 的观测。")
        summary = {
            "n": len(pair.index),
            "slope": slope,
            "intercept": intercept,
            "slope_confidence_interval": slope_interval,
            "r_squared": _finite_float(fitted.rvalue) ** 2,
            "p_value": _finite_float(fitted.pvalue),
            "alpha": alpha,
            "significant": _finite_float(fitted.pvalue) < alpha,
            "standardized_effect": {
                "name": "standardized_beta",
                "value": _finite_float(fitted.rvalue),
            },
            "diagnostics": {
                "residual_normality": normality,
                "absolute_residual_fitted_spearman": {
                    "coefficient": _finite_or_none(heteroscedasticity.statistic),
                    "p_value": variance_p,
                },
                "large_standardized_residual_count": outlier_count,
            },
        }
        return self._save_result(
            dataset_id=dataset_id,
            analysis_type="regression",
            method="simple_linear_regression",
            parameters={
                "response_column": response_column,
                "predictor_column": predictor_column,
                "alpha": alpha,
                "missing_strategy": "complete_cases",
            },
            summary=summary,
            warnings=_bounded_warnings(warnings),
            evidence_id=evidence_id,
        )

    def _save_result(
        self,
        *,
        dataset_id: str,
        analysis_type: Literal[
            "quality", "descriptive", "correlation", "statistical_test", "regression"
        ],
        method: str,
        parameters: dict[str, Any],
        summary: dict[str, Any],
        warnings: Sequence[str] = (),
        evidence_id: str | None = None,
    ) -> AnalysisResult:
        result = AnalysisResult(
            analysis_id=f"analysis-{uuid.uuid4().hex}",
            dataset_id=dataset_id,
            analysis_type=analysis_type,
            method=method,
            parameters=parameters,
            summary=summary,
            warnings=tuple(dict.fromkeys(warnings)),
            evidence_id=evidence_id or f"evidence-{uuid.uuid4().hex}",
        )
        self.store.save_analysis(result)
        return result


def _describe_series(
    series: pd.Series[Any],
    column: str,
    group: JsonScalar,
    quantiles: Sequence[float],
) -> dict[str, Any]:
    numeric = _finite_numeric(series)
    values: dict[str, Any] = {
        "column": column,
        "group": group,
        "count": len(numeric.index),
        "missing_count": int(len(series.index) - len(numeric.index)),
        "mean": _finite_or_none(numeric.mean()) if not numeric.empty else None,
        "std": _finite_or_none(numeric.std(ddof=1)) if len(numeric.index) > 1 else None,
        "min": _finite_or_none(numeric.min()) if not numeric.empty else None,
        "max": _finite_or_none(numeric.max()) if not numeric.empty else None,
        "quantiles": {},
    }
    values["quantiles"] = {
        str(value): _finite_or_none(numeric.quantile(value))
        if not numeric.empty
        else None
        for value in quantiles
    }
    return values


def _two_group_test(
    method: str,
    first: pd.Series[Any],
    second: pd.Series[Any],
    labels: list[JsonScalar],
    alpha: float,
) -> tuple[dict[str, Any], list[str]]:
    _require_sample_size(first, 2, labels[0])
    _require_sample_size(second, 2, labels[1])
    n_first, n_second = len(first.index), len(second.index)
    mean_first, mean_second = float(first.mean()), float(second.mean())
    if method in {"student_t", "welch_t"}:
        assumptions, warnings = _assumptions(
            [first, second],
            labels,
            alpha,
            require_equal_variance=method == "student_t",
        )
        equal_var = method == "student_t"
        test = stats.ttest_ind(first, second, equal_var=equal_var)
        statistic, p_value = _finite_float(test.statistic), _finite_float(test.pvalue)
        effect = _cohens_d(first, second)
        interval, degrees = _mean_difference_interval(
            first, second, alpha, equal_var=equal_var
        )
        effect_name = "cohens_d"
    elif method == "mann_whitney":
        assumptions = {"independence": "user_assumed"}
        warnings = []
        test = stats.mannwhitneyu(first, second, alternative="two-sided")
        statistic, p_value = _finite_float(test.statistic), _finite_float(test.pvalue)
        effect = 1.0 - (2.0 * statistic) / (n_first * n_second)
        interval, degrees = None, None
        effect_name = "rank_biserial_correlation"
    else:
        raise ValueError(f"unsupported two-group test: {method}")
    summary = {
        "groups": [
            {"label": labels[0], "n": n_first, "mean": mean_first},
            {"label": labels[1], "n": n_second, "mean": mean_second},
        ],
        "statistic": statistic,
        "degrees_of_freedom": degrees,
        "p_value": p_value,
        "alpha": alpha,
        "significant": p_value < alpha,
        "effect_size": {"name": effect_name, "value": effect},
        "mean_difference": mean_first - mean_second,
        "confidence_interval": interval,
        "assumptions": assumptions,
        "multiple_testing_correction": None,
    }
    return summary, warnings


def _paired_t(
    left: pd.Series[Any],
    right: pd.Series[Any],
    alpha: float,
    labels: tuple[str, str],
) -> tuple[dict[str, Any], list[str]]:
    _require_sample_size(left, 2, labels[0])
    differences = left - right
    if float(differences.std(ddof=1)) == 0:
        raise ValueError("paired differences have zero variance")
    normality = _normality_check(differences, alpha)
    warnings = [] if normality["passed"] is not False else [
        "配对差值未通过正态性检查。"
    ]
    test = stats.ttest_rel(left, right)
    statistic, p_value = _finite_float(test.statistic), _finite_float(test.pvalue)
    count = len(differences.index)
    mean_difference = float(differences.mean())
    standard_error = float(differences.std(ddof=1)) / math.sqrt(count)
    critical = float(stats.t.ppf(1 - alpha / 2, count - 1))
    interval = [
        mean_difference - critical * standard_error,
        mean_difference + critical * standard_error,
    ]
    return (
        {
            "pairs": count,
            "columns": list(labels),
            "statistic": statistic,
            "degrees_of_freedom": float(count - 1),
            "p_value": p_value,
            "alpha": alpha,
            "significant": p_value < alpha,
            "effect_size": {
                "name": "cohens_dz",
                "value": mean_difference / float(differences.std(ddof=1)),
            },
            "mean_difference": mean_difference,
            "confidence_interval": interval,
            "assumptions": {"normality_of_differences": normality},
            "multiple_testing_correction": None,
        },
        warnings,
    )


def _wilcoxon(
    left: pd.Series[Any],
    right: pd.Series[Any],
    alpha: float,
    labels: tuple[str, str],
) -> tuple[dict[str, Any], list[str]]:
    _require_sample_size(left, 2, labels[0])
    differences = left - right
    nonzero = differences[differences != 0]
    if nonzero.empty:
        raise ValueError("paired differences are all zero")
    test = stats.wilcoxon(left, right, alternative="two-sided")
    statistic = _finite_float(test.statistic)
    p_value = _finite_float(test.pvalue)
    ranks = stats.rankdata(nonzero.abs())
    positive = float(ranks[nonzero.to_numpy() > 0].sum())
    negative = float(ranks[nonzero.to_numpy() < 0].sum())
    total = positive + negative
    effect = (positive - negative) / total if total else 0.0
    return (
        {
            "pairs": len(left.index),
            "columns": list(labels),
            "statistic": statistic,
            "degrees_of_freedom": None,
            "p_value": p_value,
            "alpha": alpha,
            "significant": p_value < alpha,
            "effect_size": {
                "name": "matched_rank_biserial_correlation",
                "value": effect,
            },
            "median_difference": _finite_float(differences.median()),
            "confidence_interval": None,
            "assumptions": {"symmetry_of_differences": "user_assumed"},
            "multiple_testing_correction": None,
        },
        [],
    )


def _friedman(
    samples: list[pd.Series[Any]], labels: list[JsonScalar], alpha: float
) -> tuple[dict[str, Any], list[str]]:
    if len(samples) < 3:
        raise ValueError("friedman requires at least three repeated conditions")
    pair_count = len(samples[0].index)
    if pair_count < 2 or any(len(sample.index) != pair_count for sample in samples):
        raise ValueError("friedman requires aligned complete repeated observations")
    test = stats.friedmanchisquare(*samples)
    statistic = _finite_float(test.statistic)
    p_value = _finite_float(test.pvalue)
    effect = statistic / (pair_count * (len(samples) - 1))
    post_hoc = (
        _paired_post_hoc(samples, labels, alpha) if p_value < alpha else []
    )
    return (
        {
            "pairs": pair_count,
            "conditions": [
                {
                    "label": label,
                    "n": len(sample.index),
                    "mean": _finite_float(sample.mean()),
                    "median": _finite_float(sample.median()),
                }
                for sample, label in zip(samples, labels, strict=True)
            ],
            "statistic": statistic,
            "degrees_of_freedom": float(len(samples) - 1),
            "p_value": p_value,
            "alpha": alpha,
            "significant": p_value < alpha,
            "effect_size": {"name": "kendalls_w", "value": effect},
            "confidence_interval": None,
            "assumptions": {"complete_repeated_observations": True},
            "post_hoc": post_hoc,
            "multiple_testing_correction": "holm" if post_hoc else None,
        },
        [],
    )


def _paired_post_hoc(
    samples: list[pd.Series[Any]], labels: list[JsonScalar], alpha: float
) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    raw_p_values: list[float] = []
    for left_index in range(len(samples) - 1):
        for right_index in range(left_index + 1, len(samples)):
            left = samples[left_index]
            right = samples[right_index]
            differences = left - right
            if bool((differences == 0).all()):
                statistic, raw_p, effect = 0.0, 1.0, 0.0
            else:
                test = stats.wilcoxon(left, right, alternative="two-sided")
                statistic = _finite_float(test.statistic)
                raw_p = _finite_float(test.pvalue)
                nonzero = differences[differences != 0]
                ranks = stats.rankdata(nonzero.abs())
                positive = float(ranks[nonzero.to_numpy() > 0].sum())
                negative = float(ranks[nonzero.to_numpy() < 0].sum())
                effect = (positive - negative) / (positive + negative)
            raw_p_values.append(raw_p)
            comparisons.append(
                {
                    "condition_a": labels[left_index],
                    "condition_b": labels[right_index],
                    "method": "wilcoxon",
                    "statistic": statistic,
                    "raw_p_value": raw_p,
                    "effect_size": {
                        "name": "matched_rank_biserial_correlation",
                        "value": effect,
                    },
                }
            )
    for comparison, adjusted_p in zip(
        comparisons, _holm_adjust(raw_p_values), strict=True
    ):
        comparison["adjusted_p_value"] = adjusted_p
        comparison["significant"] = adjusted_p < alpha
    return comparisons


def _anova(
    samples: list[pd.Series[Any]], labels: list[JsonScalar], alpha: float
) -> tuple[dict[str, Any], list[str]]:
    _require_multiple_groups(samples, labels)
    assumptions, warnings = _assumptions(
        samples, labels, alpha, require_equal_variance=True
    )
    test = stats.f_oneway(*samples)
    statistic, p_value = _finite_float(test.statistic), _finite_float(test.pvalue)
    all_values = pd.concat(samples, ignore_index=True)
    grand_mean = float(all_values.mean())
    between = sum(
        len(sample.index) * (float(sample.mean()) - grand_mean) ** 2
        for sample in samples
    )
    total = float(((all_values - grand_mean) ** 2).sum())
    eta_squared = between / total if total > 0 else 0.0
    post_hoc = (
        _pairwise_post_hoc(samples, labels, alpha, method="welch_t")
        if p_value < alpha
        else []
    )
    return (
        {
            "groups": _group_summaries(samples, labels, alpha),
            "statistic": statistic,
            "degrees_of_freedom": [len(samples) - 1, len(all_values) - len(samples)],
            "p_value": p_value,
            "alpha": alpha,
            "significant": p_value < alpha,
            "effect_size": {"name": "eta_squared", "value": eta_squared},
            "confidence_interval": None,
            "assumptions": assumptions,
            "post_hoc": post_hoc,
            "multiple_testing_correction": "holm" if post_hoc else None,
        },
        warnings,
    )


def _kruskal(
    samples: list[pd.Series[Any]], labels: list[JsonScalar], alpha: float
) -> tuple[dict[str, Any], list[str]]:
    _require_multiple_groups(samples, labels)
    test = stats.kruskal(*samples)
    statistic, p_value = _finite_float(test.statistic), _finite_float(test.pvalue)
    total = sum(len(sample.index) for sample in samples)
    effect = max(0.0, (statistic - len(samples) + 1) / (total - len(samples)))
    post_hoc = (
        _pairwise_post_hoc(samples, labels, alpha, method="mann_whitney")
        if p_value < alpha
        else []
    )
    return (
        {
            "groups": _group_summaries(samples, labels, alpha),
            "statistic": statistic,
            "degrees_of_freedom": float(len(samples) - 1),
            "p_value": p_value,
            "alpha": alpha,
            "significant": p_value < alpha,
            "effect_size": {"name": "epsilon_squared", "value": effect},
            "confidence_interval": None,
            "assumptions": {"independence": "user_assumed"},
            "post_hoc": post_hoc,
            "multiple_testing_correction": "holm" if post_hoc else None,
        },
        [],
    )


def _pairwise_post_hoc(
    samples: list[pd.Series[Any]],
    labels: list[JsonScalar],
    alpha: float,
    *,
    method: Literal["welch_t", "mann_whitney"],
) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    raw_p_values: list[float] = []
    for left_index in range(len(samples) - 1):
        for right_index in range(left_index + 1, len(samples)):
            left = samples[left_index]
            right = samples[right_index]
            if method == "welch_t":
                test = stats.ttest_ind(left, right, equal_var=False)
                statistic = _finite_float(test.statistic)
                effect_name = "cohens_d"
                effect_value = _cohens_d(left, right)
                interval, _degrees = _mean_difference_interval(
                    left, right, alpha, equal_var=False
                )
            else:
                test = stats.mannwhitneyu(left, right, alternative="two-sided")
                statistic = _finite_float(test.statistic)
                effect_name = "rank_biserial_correlation"
                effect_value = 1.0 - (
                    2.0 * statistic / (len(left.index) * len(right.index))
                )
                interval = None
            raw_p = _finite_float(test.pvalue)
            raw_p_values.append(raw_p)
            comparisons.append(
                {
                    "group_a": labels[left_index],
                    "group_b": labels[right_index],
                    "method": method,
                    "statistic": statistic,
                    "raw_p_value": raw_p,
                    "effect_size": {
                        "name": effect_name,
                        "value": effect_value,
                    },
                    "difference_confidence_interval": interval,
                }
            )
    adjusted = _holm_adjust(raw_p_values)
    for comparison, adjusted_p in zip(comparisons, adjusted, strict=True):
        comparison["adjusted_p_value"] = adjusted_p
        comparison["significant"] = adjusted_p < alpha
    return comparisons


def _holm_adjust(p_values: list[float]) -> list[float]:
    count = len(p_values)
    order = sorted(range(count), key=p_values.__getitem__)
    adjusted = [0.0] * count
    running_max = 0.0
    for rank, original_index in enumerate(order):
        candidate = min(1.0, (count - rank) * p_values[original_index])
        running_max = max(running_max, candidate)
        adjusted[original_index] = running_max
    return adjusted


def _assumptions(
    samples: list[pd.Series[Any]],
    labels: list[JsonScalar],
    alpha: float,
    *,
    require_equal_variance: bool,
) -> tuple[dict[str, Any], list[str]]:
    normality = [
        {"group": label, **_normality_check(sample, alpha)}
        for sample, label in zip(samples, labels, strict=True)
    ]
    warnings = [
        f"组 {item['group']} 未通过正态性检查。"
        for item in normality
        if item["passed"] is False
    ]
    variance: dict[str, Any]
    if all(len(sample.index) >= 2 for sample in samples):
        levene = stats.levene(*samples, center="median")
        p_value = _finite_float(levene.pvalue)
        variance = {
            "method": "levene_median",
            "statistic": _finite_float(levene.statistic),
            "p_value": p_value,
            "passed": p_value >= alpha,
        }
        if p_value < alpha and require_equal_variance:
            warnings.append("各组未通过方差齐性检查。")
    else:
        variance = {"method": "levene_median", "passed": None}
    return {"normality": normality, "equal_variance": variance}, warnings


def _normality_check(sample: pd.Series[Any], alpha: float) -> dict[str, Any]:
    count = len(sample.index)
    if count < 3:
        return {"method": "shapiro", "n": count, "passed": None}
    checked = sample if count <= 5000 else sample.iloc[:5000]
    result = stats.shapiro(checked)
    p_value = _finite_float(result.pvalue)
    return {
        "method": "shapiro",
        "n": len(checked.index),
        "statistic": _finite_float(result.statistic),
        "p_value": p_value,
        "passed": p_value >= alpha,
    }


def _mean_difference_interval(
    first: pd.Series[Any],
    second: pd.Series[Any],
    alpha: float,
    *,
    equal_var: bool,
) -> tuple[list[float], float]:
    n_first, n_second = len(first.index), len(second.index)
    var_first, var_second = float(first.var(ddof=1)), float(second.var(ddof=1))
    difference = float(first.mean()) - float(second.mean())
    if equal_var:
        degrees = float(n_first + n_second - 2)
        pooled = (
            (n_first - 1) * var_first + (n_second - 1) * var_second
        ) / degrees
        standard_error = math.sqrt(pooled * (1 / n_first + 1 / n_second))
    else:
        first_term, second_term = var_first / n_first, var_second / n_second
        standard_error = math.sqrt(first_term + second_term)
        if standard_error == 0:
            raise ValueError("group values have zero pooled variance")
        denominator = (first_term**2) / (n_first - 1) + (second_term**2) / (
            n_second - 1
        )
        degrees = (first_term + second_term) ** 2 / denominator
    if standard_error == 0:
        raise ValueError("group values have zero pooled variance")
    critical = float(stats.t.ppf(1 - alpha / 2, degrees))
    return (
        [
            difference - critical * standard_error,
            difference + critical * standard_error,
        ],
        degrees,
    )


def _cohens_d(first: pd.Series[Any], second: pd.Series[Any]) -> float:
    n_first, n_second = len(first.index), len(second.index)
    pooled = math.sqrt(
        (
            (n_first - 1) * float(first.var(ddof=1))
            + (n_second - 1) * float(second.var(ddof=1))
        )
        / (n_first + n_second - 2)
    )
    if pooled == 0:
        raise ValueError("groups have zero pooled variance")
    return (float(first.mean()) - float(second.mean())) / pooled


def _paired_samples(
    frame: pd.DataFrame, columns: tuple[str, str]
) -> tuple[pd.Series[Any], pd.Series[Any]]:
    if columns[0] == columns[1]:
        raise ValueError("paired columns must be different")
    _validate_numeric_columns(frame, columns, minimum=2)
    pair = frame[list(columns)].replace([math.inf, -math.inf], pd.NA).dropna()
    return pair[columns[0]].astype(float), pair[columns[1]].astype(float)


def _repeated_samples(
    frame: pd.DataFrame,
    *,
    response_column: str,
    subject_id_column: str,
    condition_column: str,
    required_conditions: int | None = None,
    minimum_conditions: int = 2,
) -> tuple[list[pd.Series[Any]], list[JsonScalar], int]:
    _validate_numeric_columns(frame, [response_column])
    _require_column(frame, subject_id_column)
    _require_column(frame, condition_column)
    if len({response_column, subject_id_column, condition_column}) != 3:
        raise ValueError("response, subject ID and condition columns must differ")
    working = frame[
        [subject_id_column, condition_column, response_column]
    ].copy()
    working[response_column] = pd.to_numeric(
        working[response_column], errors="coerce"
    )
    working = working.dropna()
    duplicated = working.duplicated(
        subset=[subject_id_column, condition_column], keep=False
    )
    if bool(duplicated.any()):
        raise ValueError(
            "each subject-condition pair must contain exactly one observation"
        )
    labels = [_json_scalar(value) for value in working[condition_column].unique()]
    if required_conditions is not None and len(labels) != required_conditions:
        raise ValueError(
            f"paired analysis requires exactly {required_conditions} conditions"
        )
    if len(labels) < minimum_conditions or len(labels) > 20:
        raise ValueError(
            f"repeated analysis requires between {minimum_conditions} and 20 conditions"
        )
    pivot = working.pivot(
        index=subject_id_column,
        columns=condition_column,
        values=response_column,
    )
    source_subjects = len(pivot.index)
    complete = pivot.dropna(axis=0, how="any")
    if len(complete.index) < 2:
        raise ValueError("fewer than two subjects have complete repeated observations")
    samples = [
        complete[label].reset_index(drop=True).astype(float)
        for label in working[condition_column].unique()
    ]
    return samples, labels, source_subjects - len(complete.index)


def _group_samples(
    frame: pd.DataFrame,
    response_column: str,
    group_column: str,
    requested_groups: Sequence[JsonScalar] | None,
) -> tuple[list[pd.Series[Any]], list[JsonScalar]]:
    _validate_numeric_columns(frame, [response_column])
    _require_column(frame, group_column)
    source_labels = (
        list(frame[group_column].drop_duplicates())
        if requested_groups is None
        else list(requested_groups)
    )
    raw_labels = [_json_scalar(label) for label in source_labels]
    if len(raw_labels) < 2 or len(raw_labels) > 20:
        raise ValueError("statistical tests require between 2 and 20 groups")
    if len(raw_labels) != len({_label_key(label) for label in raw_labels}):
        raise ValueError("groups must be unique")
    samples: list[pd.Series[Any]] = []
    labels: list[JsonScalar] = []
    for label in raw_labels:
        mask = (
            frame[group_column].isna()
            if label is None
            else frame[group_column] == label
        )
        sample = _finite_numeric(frame.loc[mask, response_column])
        if sample.empty:
            raise ValueError(f"group {label!r} has no valid numeric observations")
        samples.append(sample)
        labels.append(_json_scalar(label))
    return samples, labels


def _validate_numeric_columns(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    minimum: int = 1,
    maximum: int = 50,
) -> tuple[str, ...]:
    selected = tuple(columns)
    if len(selected) < minimum or len(selected) > maximum:
        raise ValueError(
            f"select between {minimum} and {maximum} numeric columns"
        )
    if len(selected) != len(set(selected)):
        raise ValueError("selected columns must be unique")
    for column in selected:
        _require_column(frame, column)
        converted = pd.to_numeric(frame[column], errors="coerce")
        original_non_null = int(frame[column].notna().sum())
        if original_non_null and int(converted.notna().sum()) != original_non_null:
            raise ValueError(f"column {column!r} contains non-numeric values")
    return selected


def _validate_quantiles(values: Sequence[float]) -> tuple[float, ...]:
    quantiles = tuple(float(value) for value in values)
    if not quantiles or len(quantiles) > 9:
        raise ValueError("provide between 1 and 9 quantiles")
    if any(not 0 <= value <= 1 for value in quantiles):
        raise ValueError("quantiles must be between 0 and 1")
    if tuple(sorted(set(quantiles))) != quantiles:
        raise ValueError("quantiles must be unique and sorted")
    return quantiles


def _require_column(frame: pd.DataFrame, column: str) -> None:
    if column not in frame.columns:
        raise ValueError(f"unknown dataset column: {column!r}")


def _finite_numeric(series: pd.Series[Any]) -> pd.Series[Any]:
    numeric = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    return numeric[numeric.map(math.isfinite)]


def _is_finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _finite_or_none(value: Any) -> float | None:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _finite_float(value: Any) -> float:
    converted = _finite_or_none(value)
    if converted is None:
        raise ValueError("statistical calculation produced a non-finite result")
    return converted


def _json_scalar(value: Any) -> JsonScalar:
    if value is None or bool(pd.isna(value)):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return str(value)


def _label_key(value: JsonScalar) -> tuple[str, str]:
    return type(value).__name__, repr(value)


def _require_sample_size(
    sample: pd.Series[Any], minimum: int, label: JsonScalar
) -> None:
    if len(sample.index) < minimum:
        raise ValueError(f"group {label!r} requires at least {minimum} observations")


def _require_multiple_groups(
    samples: list[pd.Series[Any]], labels: list[JsonScalar]
) -> None:
    if len(samples) < 2:
        raise ValueError("test requires at least two groups")
    for sample, label in zip(samples, labels, strict=True):
        _require_sample_size(sample, 2, label)
    if pd.concat(samples).nunique() <= 1:
        raise ValueError("all groups contain the same constant value")


def _group_summaries(
    samples: list[pd.Series[Any]], labels: list[JsonScalar], alpha: float
) -> list[dict[str, Any]]:
    summaries = []
    for sample, label in zip(samples, labels, strict=True):
        count = len(sample.index)
        mean = float(sample.mean())
        standard_error = float(sample.std(ddof=1)) / math.sqrt(count)
        critical = float(stats.t.ppf(1 - alpha / 2, count - 1))
        summaries.append(
            {
                "label": label,
                "n": count,
                "mean": mean,
                "mean_confidence_interval": [
                    mean - critical * standard_error,
                    mean + critical * standard_error,
                ],
            }
        )
    return summaries


def _bounded_warnings(values: Sequence[str]) -> tuple[str, ...]:
    unique = list(dict.fromkeys(values))
    if len(unique) <= 64:
        return tuple(unique)
    omitted = len(unique) - 63
    return (*unique[:63], f"另有 {omitted} 条重复模式警告未逐项显示。")
