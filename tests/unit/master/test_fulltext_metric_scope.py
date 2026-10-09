"""Requested metrics, not general research goals, authorize analysis rows."""

import pytest

from materials_screening.master.fulltext_analysis import FulltextAnalysisProcessor
from materials_screening.master.fulltext_analysis_contracts import (
    AnalysisScopePlan,
    MetricScope,
)


def scope(name, quote, ids=()):
    return AnalysisScopePlan(
        metrics=(MetricScope(name=name, task_quote=quote, measurement_ids=ids),),
        requested_operations=("describe",),
    )


def test_repair_feedback_lists_each_unsupported_metric_and_exact_task_quote():
    from materials_screening.master.fulltext_analysis import _scope_metric_feedback

    goal = "判断ZnO是否适合紫外探测器。"
    request = "全文提取响应度、增益及光照和偏压条件。"
    proposal = AnalysisScopePlan(
        metrics=(
            MetricScope(name="Bandgap", task_quote=goal),
            MetricScope(name="Cutoff Wavelength", task_quote=request),
            MetricScope(name="Responsivity", task_quote=request),
        ),
        requested_operations=("describe",),
    )
    feedback = _scope_metric_feedback(proposal, {}, goal + request)
    assert [(r["metric_name"], r["task_quote"]) for r in feedback["issues"]] == [
        ("Bandgap", goal),
        ("Cutoff Wavelength", request),
    ]
    assert feedback["supported_proposal_metrics"] == ["Responsivity"]
    assert all(r["action"] == "remove_scope_entry" for r in feedback["issues"])
    assert _scope_metric_feedback(None, {}, goal) == {}


@pytest.mark.parametrize(
    "metric",
    ["Bandgap", "Turn-on Voltage", "Cutoff Wavelength", "Fall Time Under UV Light"],
)
def test_general_goal_or_other_request_does_not_authorize_extra_metric(metric):
    requirements = "判断ZnO是否适合紫外探测器。全文提取响应度、增益及光照和偏压条件。"
    for quote in (
        "判断ZnO是否适合紫外探测器。",
        "全文提取响应度、增益及光照和偏压条件。",
    ):
        with pytest.raises(ValueError, match="explicitly requested"):
            FulltextAnalysisProcessor._validate_scope(
                scope(metric, quote), {}, requirements
            )


def test_matching_label_cannot_hide_unrequested_row():
    quote = "全文提取响应度。"
    with pytest.raises(ValueError, match="does not match"):
        FulltextAnalysisProcessor._validate_scope(
            scope("Responsivity", quote, ("x",)), {"x": {"metric": "Bandgap"}}, quote
        )


@pytest.mark.parametrize(
    ("name", "quote"),
    [
        ("Responsivity", "全文提取响应度。"),
        ("Gain", "全文提取增益。"),
        ("Bandgap", "全文提取带隙。"),
        ("novel metric", "Extract novel metric from the paper."),
    ],
)
def test_explicit_metrics_are_not_limited_to_zno(name, quote):
    candidates = {"x": {"metric": name}}
    FulltextAnalysisProcessor._validate_scope(
        scope(name, quote, ("x",)), candidates, quote
    )
    assert candidates == {"x": {"metric": name}}
