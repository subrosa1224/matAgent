from __future__ import annotations

from pathlib import Path

from materials_screening.data_analysis.dataset_store import DatasetStore
from materials_screening.data_analysis.interpretation import (
    ScientificInterpretationService,
    render_scientific_interpretation,
)
from materials_screening.data_analysis.scientific import ScientificAnalysisBrief
from materials_screening.data_analysis.statistics import DataStatisticsService


def test_significant_anova_becomes_bounded_research_narrative(
    tmp_path: Path,
) -> None:
    store = DatasetStore(tmp_path / "store")
    dataset = store.register_file(
        Path("examples/data_analysis/grouped_experiment_demo.csv"),
        source_artifact_id="artifact-interpretation-anova",
    )
    result = DataStatisticsService(store).run_statistical_test(
        dataset.dataset_id,
        method="anova",
        response_column="hardness_hv",
        group_column="method",
    )
    brief = ScientificAnalysisBrief(
        dataset_id=dataset.dataset_id,
        domain="materials_science",
        research_question="不同制备方法组的材料硬度是否存在差异？",
        observation_unit="独立材料试样",
        design="independent_groups",
        response_variables=("hardness_hv",),
        group_variable="method",
        practical_thresholds={"hardness_hv": 20.0},
        units={"hardness_hv": "HV"},
        roles_confirmed=True,
    )

    interpretation = ScientificInterpretationService(store).interpret(
        brief, result
    )
    rendered = render_scientific_interpretation(interpretation)

    assert interpretation.evidence_strength == "较强"
    assert "B（581.1 HV） > A（540.6 HV） > C（521 HV）" in rendered
    assert "B 比 A 高 40.5 HV" in rendered
    assert "达到该阈值" in interpretation.practical_significance
    assert "不能验证随机分组" in rendered
    assert "不直接宣称处理造成了变化" in rendered
    assert "不能据此证明完全正态" in rendered
    assert "temperature_c" in rendered
    assert result.analysis_id not in rendered


def test_non_significant_result_uses_insufficient_evidence_not_equality(
    tmp_path: Path,
) -> None:
    source = tmp_path / "nonsignificant.csv"
    source.write_text(
        "sample_id,group,value\n"
        "A1,A,1\nA2,A,2\nA3,A,3\nA4,A,4\nA5,A,5\n"
        "B1,B,1.1\nB2,B,2.1\nB3,B,3.1\nB4,B,4.1\nB5,B,5.1\n",
        encoding="utf-8",
    )
    store = DatasetStore(tmp_path / "store")
    dataset = store.register_file(
        source, source_artifact_id="artifact-interpretation-null"
    )
    result = DataStatisticsService(store).run_statistical_test(
        dataset.dataset_id,
        method="welch_t",
        response_column="value",
        group_column="group",
    )
    brief = ScientificAnalysisBrief(
        dataset_id=dataset.dataset_id,
        research_question="A、B两组的指标是否存在差异？",
        observation_unit="独立样本",
        design="independent_groups",
        response_variables=("value",),
        group_variable="group",
        roles_confirmed=True,
    )

    interpretation = ScientificInterpretationService(store).interpret(
        brief, result
    )

    assert interpretation.evidence_strength == "有限"
    assert "没有提供足够证据" in interpretation.direct_answer
    assert "不等同于证明" in interpretation.direct_answer
    assert "完全相同" in interpretation.direct_answer
