"""Supplementary evidence cannot replace the requested outcome."""

import pytest

from materials_screening.master.fulltext_analysis import (
    FulltextAnalysisProcessor,
    _scope_metric_feedback,
)
from materials_screening.master.fulltext_analysis_contracts import (
    AnalysisScopePlan,
    MetricScope,
)

QUOTE = "全文提取降解率。"


def proposal(**updates):
    return AnalysisScopePlan(
        metrics=(
            MetricScope(name="Degradation", task_quote=QUOTE, measurement_ids=("d",)),
            MetricScope(
                name="TOC",
                task_quote=QUOTE,
                measurement_ids=("t",),
                role="supplementary",
                supplementary_reason="补充考察矿化程度",
                **updates,
            ),
        ),
        requested_operations=("describe",),
    )


CANDIDATES = {
    "d": {"metric": "Degradation", "unit": "%"},
    "t": {"metric": "TOC", "unit": "%"},
}


def test_requested_plus_source_matched_supplement_is_valid():
    scope = proposal()
    FulltextAnalysisProcessor._validate_scope(scope, CANDIDATES, QUOTE)
    assert _scope_metric_feedback(scope, CANDIDATES, QUOTE)["issues"] == []


def test_supplement_cannot_replace_requested_candidate():
    scope = proposal().model_copy(update={"metrics": (proposal().metrics[1],)})
    with pytest.raises(ValueError, match="omitted"):
        FulltextAnalysisProcessor._validate_scope(scope, CANDIDATES, QUOTE)


@pytest.mark.parametrize(
    "changes",
    [
        {"requested_metric": "降解率"},
        {"measurement_ids": ()},
        {"supplementary_reason": " "},
        {"name": "Bandgap"},
        {"task_quote": "编造的用户要求"},
        {"task_quote": "从 Materials Project 查询计算带隙"},
    ],
)
def test_invalid_supplement_is_rejected(changes):
    scope = proposal()
    metric = scope.metrics[1].model_copy(update=changes)
    scope = scope.model_copy(update={"metrics": (scope.metrics[0], metric)})
    with pytest.raises(ValueError):
        FulltextAnalysisProcessor._validate_scope(
            scope, CANDIDATES, QUOTE + "从 Materials Project 查询计算带隙"
        )


def test_supplement_role_and_purpose_are_visible_in_actual_report(tmp_path):
    from tests.unit.master.test_fulltext_analysis import analysis_env
    from tests.unit.master.test_fulltext_preview import run

    def configure(env):
        import json

        from materials_screening.master.fulltext_scope_stages import (
            SupplementaryScopePlan,
        )

        env.scope_quote = "全文提取增益并分析辅助证据。"
        env.scope_metric = "Gain"

        def supplement(kwargs):
            rows = json.loads(kwargs["user_text"])["candidate_metadata"]
            return SupplementaryScopePlan(
                metrics=(
                    MetricScope(
                        name=rows[0]["metric"],
                        task_quote=env.scope_quote,
                        measurement_ids=tuple(row["measurement_id"] for row in rows),
                        role="supplementary",
                        supplementary_reason="提供强度背景证据",
                    ),
                )
            )

        env.supplement_plan = supplement

    env = analysis_env(tmp_path, configure=configure)
    result = run(env, env.task)
    assert result.status == "completed"
    assert "补充指标（不替代主要任务）" in result.response_text
    assert "提供强度背景证据" in result.response_text
    assert "| 补充 |" in result.response_text
    assert "不代表主要任务已完成" in result.response_text
