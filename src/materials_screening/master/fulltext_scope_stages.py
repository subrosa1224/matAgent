"""Disjoint scope-stage inputs; auxiliary output cannot rewrite primary intent."""

from pydantic import BaseModel, ConfigDict, Field

from .fulltext_analysis_contracts import AnalysisScopePlan, MetricScope


class SupplementaryScopePlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    metrics: tuple[MetricScope, ...] = Field(default=(), max_length=20)


PRIMARY_STAGE_PROMPT = """
This call is PRIMARY MATCHING ONLY. Return a complete AnalysisScopePlan with
metrics and requested_operations; all metrics must have role="requested".
Only supplied primary candidate IDs can be selected or excluded. Do not choose
TOC, bandgap or other ancillary quantities to stand in for a requested outcome.
If the requested outcome is unavailable, keep its explicitly requested metric
with empty measurement_ids and explain the limitation. No supplementary entries.
Return the plan itself, not the input envelope or feedback.
"""

SUPPLEMENT_STAGE_PROMPT = """Choose useful supplementary evidence ONLY from the
supplied remaining candidate_metadata. The validated primary plan is READ ONLY.
Return exactly {"metrics": [...]} or {"metrics": []} when no relevant supplement.
Do not return input, primary_scope, feedback, operations, conditions or exclusions.
Each metric needs its exact source name, role="supplementary", requested_metric=null,
measurement_ids from the remaining candidates, task_quote copied verbatim from
requirements and supplementary_reason explaining relevance. Do not invent values.
Never relabel TOC as degradation or wavelength as responsivity. Supplemental
selection cannot replace missing primary outcomes or authorize ranking/trends.
Source validity is checked; scientific relevance remains pending expert review.
"""


def primary_candidates(candidates, requirements):
    from .fulltext_analysis import (
        _expected_candidate_ids,
        _experimental_requirements,
        _metric_named_in_quote,
    )

    expected = _expected_candidate_ids(candidates, requirements)
    clauses = _experimental_requirements(requirements)
    return {
        mid: row
        for mid, row in candidates.items()
        if mid in expected
        or any(_metric_named_in_quote(row["metric"], quote) for quote in clauses)
    }


def supplementary_candidates(candidates, primary_pool, primary):
    accounted = set(primary_pool) | set(primary.excluded_measurements)
    accounted.update(
        mid for metric in primary.metrics for mid in metric.measurement_ids
    )
    return {mid: row for mid, row in candidates.items() if mid not in accounted}


def merge_scopes(primary, supplement):
    if any(metric.role != "requested" for metric in primary.metrics):
        raise ValueError("Primary stage cannot choose supplementary metrics")
    if any(metric.role != "supplementary" for metric in supplement.metrics):
        raise ValueError("Supplementary stage cannot rewrite primary metrics")
    return AnalysisScopePlan.model_validate(
        {**primary.model_dump(), "metrics": (*primary.metrics, *supplement.metrics)}
    )
