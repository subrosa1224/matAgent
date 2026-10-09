"""Primary matching and auxiliary selection have disjoint inputs."""

import pytest

from materials_screening.master.fulltext_analysis_contracts import (
    AnalysisScopePlan,
    MetricScope,
)
from materials_screening.master.fulltext_scope_stages import (
    SupplementaryScopePlan,
    merge_scopes,
    primary_candidates,
    supplementary_candidates,
)


def test_candidate_pools_separate_degradation_from_toc():
    rows = {
        "d": {"metric": "Degradation", "unit": "%"},
        "t": {"metric": "TOC", "unit": "%"},
    }
    primary = primary_candidates(rows, "全文提取降解率。")
    assert set(primary) == {"d"}
    plan = AnalysisScopePlan(
        metrics=(
            MetricScope(
                name="Degradation",
                task_quote="全文提取降解率。",
                measurement_ids=("d",),
            ),
        ),
        requested_operations=("describe",),
    )
    assert set(supplementary_candidates(rows, primary, plan)) == {"t"}


def test_supplement_merge_cannot_change_primary_operations_or_exclusions():
    primary = AnalysisScopePlan(
        metrics=(MetricScope(name="增益", task_quote="提取增益"),),
        requested_operations=("trend",),
        excluded_measurements={"x": "非目标对照"},
    )
    supplement = SupplementaryScopePlan(
        metrics=(
            MetricScope(
                name="TOC",
                task_quote="提取增益",
                measurement_ids=("t",),
                role="supplementary",
                supplementary_reason="背景",
            ),
        )
    )
    combined = merge_scopes(primary, supplement)
    assert combined.metrics[0] == primary.metrics[0]
    assert combined.requested_operations == primary.requested_operations
    assert combined.excluded_measurements == primary.excluded_measurements


def test_primary_role_is_not_accepted_in_supplement_stage():
    primary = AnalysisScopePlan(
        metrics=(MetricScope(name="增益", task_quote="提取增益"),),
        requested_operations=("describe",),
    )
    with pytest.raises(ValueError):
        merge_scopes(primary, SupplementaryScopePlan(metrics=primary.metrics))


def test_failed_supplement_preserves_primary_checkpoint_and_resume(tmp_path):
    from materials_screening.llm.errors import LLMStructuredOutputError
    from tests.unit.master.test_fulltext_analysis import analysis_env
    from tests.unit.master.test_fulltext_preview import run

    def configure(env):
        env.scope_quote = "全文提取增益。"
        env.scope_metric = "Gain"

        def invalid(_kwargs):
            raise LLMStructuredOutputError("Invalid JSON")

        env.supplement_plan = invalid

    env = analysis_env(tmp_path, configure=configure)
    result = run(env, env.task)
    task = env.saved[-1]
    assert result.error["code"] == "FULLTEXT_ANALYSIS_FAILED"
    record = env.analysis_store.load(task.fulltext_analysis_ref)
    assert record.primary_scope is not None and record.scope is None
    saved_primary = record.primary_scope
    assert env.scope_calls == 1
    env.supplement_plan = lambda _kwargs: SupplementaryScopePlan()
    result = run(env, task)
    resumed = env.analysis_store.load(env.saved[-1].fulltext_analysis_ref)
    assert result.status == "completed"
    assert resumed.primary_scope == saved_primary
    assert env.scope_calls == 1
    assert "缺项" in result.response_text
