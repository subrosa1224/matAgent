"""Strict task-bound analysis checkpoints, separate from extraction snapshots."""

from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from .figure_evidence_contracts import FigureBatchReference
from .fulltext_condition_locator import ConditionLocations
from .fulltext_conditions import ConditionPlan
from .fulltext_snapshots import SnapshotReference
from .fulltext_targeted_evidence import TargetedEvidencePlan


class MetricScope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str = Field(min_length=1, max_length=160)
    role: Literal["requested", "supplementary"] = "requested"
    supplementary_reason: str | None = Field(default=None, max_length=1000)
    requested_metric: str | None = Field(
        default=None,
        min_length=1,
        max_length=160,
        description="Metric explicitly requested in task_quote, preferably "
        "its exact short phrase (e.g. 降解率). Do not copy a verbose source "
        "name absent from the user request.",
    )
    task_quote: str = Field(min_length=1, max_length=2000)
    measurement_ids: tuple[str, ...] = Field(default=(), max_length=1000)


class AnalysisScopePlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    metrics: tuple[MetricScope, ...] = Field(min_length=1, max_length=20)
    requested_operations: tuple[Literal["describe", "compare", "trend"], ...] = Field(
        min_length=1, max_length=3
    )
    required_conditions: tuple[str, ...] = Field(default=(), max_length=20)
    independent_variable: str | None = Field(default=None, max_length=100)
    limitations: tuple[str, ...] = Field(default=(), max_length=20)
    excluded_measurements: dict[str, str] = Field(default_factory=dict, max_length=1000)


class FulltextAnalysisReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    record_id: str = Field(pattern=r"^fulltext-analysis-[a-f0-9]{32}$")
    content_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class FulltextAnalysisRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    record_id: str = Field(
        default_factory=lambda: "fulltext-analysis-" + uuid4().hex,
        pattern=r"^fulltext-analysis-[a-f0-9]{32}$",
    )
    parent_record_id: str | None = Field(
        default=None, pattern=r"^fulltext-analysis-[a-f0-9]{32}$"
    )
    task_id: str = Field(pattern=r"^task-fulltext-[a-f0-9]{32}$")
    conversation_id: str
    policy_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    selected_snapshots: dict[str, SnapshotReference]
    scope: AnalysisScopePlan | None = None
    primary_scope: AnalysisScopePlan | None = None
    dataset_ids: dict[str, str] = Field(default_factory=dict)
    analysis_ids: dict[str, str] = Field(default_factory=dict)
    analysis_digests: dict[str, str] = Field(default_factory=dict)
    query_digests: dict[str, str] = Field(default_factory=dict)
    scope_exclusions: dict[str, str] = Field(default_factory=dict)
    condition_plans: dict[str, ConditionPlan] = Field(default_factory=dict)
    condition_locations: dict[str, ConditionLocations] = Field(default_factory=dict)
    targeted_plans: dict[str, TargetedEvidencePlan] = Field(default_factory=dict)
    figure_evidence_ref: FigureBatchReference | None = None
    annotation_format: Literal["inline-v1", "refs-v2"] = "inline-v1"
    status: Literal["collecting", "complete"] = "collecting"
    scientific_coverage: Literal["partial"] = "partial"
    report_markdown: str = Field(default="", max_length=240000)
    warnings: tuple[str, ...] = ()
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
