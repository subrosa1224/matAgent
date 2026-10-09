"""Model-proposed names require bounded physical-quantity checks."""

import pytest

from materials_screening.master.fulltext_analysis import FulltextAnalysisProcessor
from materials_screening.master.fulltext_analysis_contracts import (
    AnalysisScopePlan,
    MetricScope,
)

LONG = "Photocatalytic degradation efficiency of TC under visible light for 3 h"


def plan(name, requested, quote):
    return AnalysisScopePlan(
        metrics=(
            MetricScope(
                name=name,
                requested_metric=requested,
                task_quote=quote,
                measurement_ids=("x",),
            ),
        ),
        requested_operations=("describe",),
    )


def test_long_degradation_mapping_preserves_source():
    row = {"metric": LONG, "unit": "%", "value_text": "90.1"}
    before = row.copy()
    quote = "全文提取降解率。"
    FulltextAnalysisProcessor._validate_scope(
        plan(LONG, "降解率", quote), {"x": row}, quote
    )
    assert row == before


@pytest.mark.parametrize(
    ("name", "unit", "requested", "quote"),
    [
        ("Responsivity Peak Sensitivity", "nm", "响应度", "全文提取响应度。"),
        ("Responsivity", "nm", "响应度", "全文提取响应度。"),
        ("TOC concentration after degradation", "%", "降解率", "全文提取降解率。"),
        ("Degradation rate constant", "min−1", "降解率", "全文提取降解率。"),
        (LONG, "min−1", "降解率", "全文提取降解率。"),
        (LONG, "%", "降解率", "判断材料是否值得研究。"),
    ],
)
def test_wrong_physical_quantity_or_missing_request_is_rejected(
    name, unit, requested, quote
):
    with pytest.raises(ValueError):
        FulltextAnalysisProcessor._validate_scope(
            plan(name, requested, quote), {"x": {"metric": name, "unit": unit}}, quote
        )


def test_legacy_metric_scope_still_loads_without_mapping():
    scope = MetricScope(name="Gain", task_quote="全文提取增益。")
    assert scope.requested_metric is None


def test_feedback_corrects_mapping_instead_of_dropping_requested_long_metric():
    from materials_screening.master.fulltext_analysis import _scope_metric_feedback

    quote = "全文提取降解率。"
    proposal = plan(LONG, LONG, quote)
    feedback = _scope_metric_feedback(
        proposal, {"x": {"metric": LONG, "unit": "%"}}, quote
    )
    assert feedback["issues"][0]["action"] == "set_requested_metric"
    assert feedback["issues"][0]["proposed_requested_metrics"] == ["降解率"]
    with pytest.raises(ValueError):
        FulltextAnalysisProcessor._validate_scope(
            proposal, {"x": {"metric": LONG, "unit": "%"}}, quote
        )
