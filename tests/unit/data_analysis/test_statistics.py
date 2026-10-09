from __future__ import annotations

import math
from pathlib import Path

import pytest

from materials_screening.data_analysis.dataset_store import DatasetStore
from materials_screening.data_analysis.statistics import DataStatisticsService


def _statistics_service(
    tmp_path: Path,
) -> tuple[DataStatisticsService, DatasetStore, str]:
    source = tmp_path / "statistics.csv"
    source.write_text(
        "group,value,x,y,z,constant\n"
        "A,1,1,2,9,1\n"
        "A,2,2,4,8,1\n"
        "A,3,3,6,7,1\n"
        "B,4,4,8,6,1\n"
        "B,5,5,10,5,1\n"
        "B,6,6,12,4,1\n"
        "C,7,7,14,3,1\n"
        "C,8,8,16,2,1\n"
        "C,9,9,18,1,1\n",
        encoding="utf-8",
    )
    store = DatasetStore(tmp_path / "private")
    reference = store.register_file(
        source, source_artifact_id="artifact-data-statistics"
    )
    return DataStatisticsService(store), store, reference.dataset_id


def test_describe_dataset_returns_overall_and_grouped_statistics(
    tmp_path: Path,
) -> None:
    service, store, dataset_id = _statistics_service(tmp_path)

    overall = service.describe_dataset(dataset_id, columns=["value"])
    grouped = service.describe_dataset(
        dataset_id, columns=["value"], group_by="group"
    )

    statistic = overall.summary["statistics"][0]
    assert statistic["count"] == 9
    assert statistic["mean"] == pytest.approx(5.0)
    assert statistic["std"] == pytest.approx(math.sqrt(7.5))
    assert statistic["quantiles"]["0.5"] == pytest.approx(5.0)
    groups = grouped.summary["statistics"]
    assert [item["group"] for item in groups] == ["A", "B", "C"]
    assert [item["mean"] for item in groups] == pytest.approx([2.0, 5.0, 8.0])
    assert store.get_analysis(overall.analysis_id) == overall


@pytest.mark.parametrize("method", ["pearson", "spearman"])
def test_correlation_returns_pairwise_counts_and_coefficients(
    tmp_path: Path, method: str
) -> None:
    service, _, dataset_id = _statistics_service(tmp_path)

    result = service.analyze_correlations(
        dataset_id,
        columns=["x", "y", "z"],
        method=method,  # type: ignore[arg-type]
    )

    pairs = {(item["x"], item["y"]): item for item in result.summary["pairs"]}
    assert pairs[("x", "x")]["coefficient"] == 1.0
    assert pairs[("x", "y")]["coefficient"] == pytest.approx(1.0)
    assert pairs[("x", "z")]["coefficient"] == pytest.approx(-1.0)
    assert pairs[("x", "z")]["n"] == 9


def test_correlation_warns_for_constant_column(tmp_path: Path) -> None:
    service, _, dataset_id = _statistics_service(tmp_path)

    result = service.analyze_correlations(
        dataset_id, columns=["x", "constant"]
    )

    assert result.warnings
    pairs = result.summary["pairs"]
    assert any(item["coefficient"] is None for item in pairs)


@pytest.mark.parametrize(
    ("method", "expected_statistic", "expected_effect"),
    [
        ("student_t", -3.674234614, -3.0),
        ("welch_t", -3.674234614, -3.0),
        ("mann_whitney", 0.0, 1.0),
    ],
)
def test_two_group_tests_match_reference_values(
    tmp_path: Path,
    method: str,
    expected_statistic: float,
    expected_effect: float,
) -> None:
    service, _, dataset_id = _statistics_service(tmp_path)

    result = service.run_statistical_test(
        dataset_id,
        method=method,  # type: ignore[arg-type]
        response_column="value",
        group_column="group",
        groups=["A", "B"],
    )

    assert result.summary["statistic"] == pytest.approx(expected_statistic)
    assert result.summary["effect_size"]["value"] == pytest.approx(expected_effect)
    assert result.summary["groups"][0]["n"] == 3
    assert result.summary["groups"][1]["n"] == 3
    expected_p = 0.1 if method == "mann_whitney" else 0.021311641
    assert result.summary["p_value"] == pytest.approx(expected_p)


def test_anova_and_kruskal_return_omnibus_effect_sizes(tmp_path: Path) -> None:
    service, _, dataset_id = _statistics_service(tmp_path)

    anova = service.run_statistical_test(
        dataset_id,
        method="anova",
        response_column="value",
        group_column="group",
    )
    kruskal = service.run_statistical_test(
        dataset_id,
        method="kruskal_wallis",
        response_column="value",
        group_column="group",
    )

    assert anova.summary["statistic"] == pytest.approx(27.0)
    assert anova.summary["effect_size"]["name"] == "eta_squared"
    assert anova.summary["effect_size"]["value"] == pytest.approx(0.9)
    assert anova.summary["multiple_testing_correction"] == "holm"
    assert len(anova.summary["post_hoc"]) == 3
    assert all(
        item["adjusted_p_value"] >= item["raw_p_value"]
        for item in anova.summary["post_hoc"]
    )
    assert kruskal.summary["effect_size"]["name"] == "epsilon_squared"
    assert kruskal.summary["degrees_of_freedom"] == 2.0
    assert kruskal.summary["multiple_testing_correction"] == "holm"
    assert len(kruskal.summary["post_hoc"]) == 3


def test_paired_t_uses_complete_pairs_and_reports_dz(tmp_path: Path) -> None:
    source = tmp_path / "paired.csv"
    source.write_text(
        "before,after\n1,2\n2,4\n3,6\n4,8\n5,10\n6,\n",
        encoding="utf-8",
    )
    store = DatasetStore(tmp_path / "private")
    reference = store.register_file(
        source, source_artifact_id="artifact-data-paired"
    )

    result = DataStatisticsService(store).run_statistical_test(
        reference.dataset_id,
        method="paired_t",
        paired_columns=("before", "after"),
    )

    assert result.summary["pairs"] == 5
    assert result.summary["statistic"] == pytest.approx(-4.242640687)
    assert result.summary["p_value"] == pytest.approx(0.0132355996)
    assert result.summary["effect_size"]["value"] == pytest.approx(-1.897366596)
    assert len(result.summary["confidence_interval"]) == 2


def test_long_format_paired_tests_align_subjects_and_drop_incomplete(
    tmp_path: Path,
) -> None:
    source = tmp_path / "paired-long.csv"
    source.write_text(
        "sample,condition,value\n"
        "S1,before,10\nS1,after,11\n"
        "S2,before,10\nS2,after,12\n"
        "S3,before,10\nS3,after,13\n"
        "S4,before,11\nS4,after,13\n"
        "S5,before,10\nS5,after,14\n"
        "S6,before,12\n",
        encoding="utf-8",
    )
    store = DatasetStore(tmp_path / "private")
    reference = store.register_file(
        source, source_artifact_id="artifact-data-paired-long"
    )
    service = DataStatisticsService(store)

    paired = service.run_statistical_test(
        reference.dataset_id,
        method="paired_t",
        response_column="value",
        subject_id_column="sample",
        condition_column="condition",
    )
    wilcoxon = service.run_statistical_test(
        reference.dataset_id,
        method="wilcoxon",
        response_column="value",
        subject_id_column="sample",
        condition_column="condition",
    )

    assert paired.summary["pairs"] == 5
    assert paired.summary["dropped_incomplete_subjects"] == 1
    assert paired.summary["effect_size"]["name"] == "cohens_dz"
    assert wilcoxon.summary["pairs"] == 5
    assert wilcoxon.summary["effect_size"]["name"] == (
        "matched_rank_biserial_correlation"
    )


def test_friedman_reports_kendalls_w_and_holm_post_hoc(tmp_path: Path) -> None:
    source = tmp_path / "repeated.csv"
    rows = ["sample,time,value"]
    for sample in range(1, 7):
        rows.extend(
            (
                f"S{sample},T1,{sample}",
                f"S{sample},T2,{sample + 2}",
                f"S{sample},T3,{sample + 5}",
            )
        )
    source.write_text("\n".join(rows) + "\n", encoding="utf-8")
    store = DatasetStore(tmp_path / "private")
    reference = store.register_file(
        source, source_artifact_id="artifact-data-repeated"
    )

    result = DataStatisticsService(store).run_statistical_test(
        reference.dataset_id,
        method="friedman",
        response_column="value",
        subject_id_column="sample",
        condition_column="time",
    )

    assert result.summary["pairs"] == 6
    assert result.summary["effect_size"]["name"] == "kendalls_w"
    assert result.summary["effect_size"]["value"] == pytest.approx(1.0)
    assert result.summary["multiple_testing_correction"] == "holm"
    assert len(result.summary["post_hoc"]) == 3


def test_simple_linear_regression_reports_interval_effect_and_diagnostics(
    tmp_path: Path,
) -> None:
    source = tmp_path / "regression.csv"
    source.write_text(
        "temperature,hardness\n"
        "1,2.1\n2,4.2\n3,5.8\n4,8.3\n5,9.7\n6,12.2\n",
        encoding="utf-8",
    )
    store = DatasetStore(tmp_path / "private")
    reference = store.register_file(
        source, source_artifact_id="artifact-data-regression"
    )

    result = DataStatisticsService(store).run_linear_regression(
        reference.dataset_id,
        response_column="hardness",
        predictor_column="temperature",
    )

    assert result.analysis_type == "regression"
    assert result.summary["n"] == 6
    assert result.summary["slope"] == pytest.approx(2.0, rel=0.05)
    assert result.summary["r_squared"] > 0.99
    assert len(result.summary["slope_confidence_interval"]) == 2
    assert "residual_normality" in result.summary["diagnostics"]


def test_describe_omits_non_finite_values_and_records_missing_count(
    tmp_path: Path,
) -> None:
    source = tmp_path / "nonfinite.csv"
    source.write_text("value\n1\n2\ninf\n-inf\n", encoding="utf-8")
    store = DatasetStore(tmp_path / "private")
    reference = store.register_file(
        source, source_artifact_id="artifact-data-nonfinite"
    )

    result = DataStatisticsService(store).describe_dataset(
        reference.dataset_id, columns=["value"]
    )

    statistic = result.summary["statistics"][0]
    assert statistic["count"] == 2
    assert statistic["missing_count"] == 2
    assert statistic["mean"] == 1.5


def test_statistics_reject_invalid_columns_parameters_and_degenerate_data(
    tmp_path: Path,
) -> None:
    service, _, dataset_id = _statistics_service(tmp_path)

    with pytest.raises(ValueError, match="unknown dataset column"):
        service.describe_dataset(dataset_id, columns=["missing"])
    with pytest.raises(ValueError, match="quantiles"):
        service.describe_dataset(dataset_id, columns=["x"], quantiles=[0.5, 0.25])
    with pytest.raises(ValueError, match="between 0 and 1"):
        service.run_statistical_test(
            dataset_id,
            method="anova",
            response_column="value",
            group_column="group",
            alpha=1.0,
        )
    with pytest.raises(ValueError, match="constant value"):
        service.run_statistical_test(
            dataset_id,
            method="anova",
            response_column="constant",
            group_column="group",
        )


def test_analysis_store_rejects_unknown_or_unsafe_analysis_id(tmp_path: Path) -> None:
    _, store, _ = _statistics_service(tmp_path)

    with pytest.raises(KeyError, match="unknown analysis"):
        store.get_analysis("analysis-unknown")
    with pytest.raises(ValueError, match="path is invalid"):
        store.get_analysis("../../secret")
