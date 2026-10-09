"""Strict tool contracts for the data-analysis sub-agent."""

from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from materials_screening.data_analysis.models import (
    AnalysisResult,
    DataAnalysisArtifact,
    DatasetInspection,
    DatasetReference,
    DatasetTransformOperation,
    DatasetTransformRecord,
    JsonScalar,
)


class InspectDatasetInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    dataset_id: str
    offset: int = Field(default=0, ge=0, le=100_000)
    limit: int = Field(default=20, ge=1, le=100)


class AssessDataQualityInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    dataset_id: str


class DescribeDatasetInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    dataset_id: str
    columns: tuple[str, ...] = Field(min_length=1, max_length=50)
    group_by: str | None = None
    quantiles: tuple[float, ...] = (0.25, 0.5, 0.75)


class AnalyzeCorrelationsInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    dataset_id: str
    columns: tuple[str, ...] = Field(min_length=2, max_length=40)
    method: Literal["pearson", "spearman"] = "pearson"


class RunStatisticalTestInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    dataset_id: str
    method: Literal[
        "student_t",
        "welch_t",
        "paired_t",
        "mann_whitney",
        "anova",
        "kruskal_wallis",
    ]
    response_column: str | None = None
    group_column: str | None = None
    groups: tuple[JsonScalar, ...] | None = Field(default=None, max_length=20)
    paired_columns: tuple[str, str] | None = None
    alpha: float = Field(default=0.05, gt=0, lt=1)


class TransformDatasetInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    dataset_id: str
    operations: tuple[DatasetTransformOperation, ...] = Field(
        min_length=1, max_length=32
    )


class CreateAnalysisPlotInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    dataset_id: str
    plot_type: Literal["histogram", "boxplot", "scatter", "line", "bar", "heatmap"]
    x: str | None = None
    y: str | None = None
    group_by: str | None = None
    columns: tuple[str, ...] = Field(default=(), max_length=20)
    title: str | None = Field(default=None, max_length=256)
    x_label: str | None = Field(default=None, max_length=256)
    y_label: str | None = Field(default=None, max_length=256)
    correlation_method: Literal["pearson", "spearman"] = "pearson"


class CreateAnalysisReportInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    dataset_id: str
    output_kind: Literal["analysis_report", "dataset_export"] = "analysis_report"
    analysis_ids: tuple[str, ...] = Field(default=(), max_length=100)
    plot_artifact_ids: tuple[str, ...] = Field(default=(), max_length=50)
    format: Literal["docx", "md", "csv", "json"] = "md"

    @model_validator(mode="after")
    def validate_mode(self) -> Self:
        if self.output_kind == "analysis_report" and not self.analysis_ids:
            raise ValueError("analysis_report requires analysis_ids")
        if self.output_kind == "dataset_export":
            if self.analysis_ids or self.plot_artifact_ids:
                raise ValueError("dataset_export does not accept analysis references")
            if self.format in {"md", "docx"}:
                raise ValueError("dataset_export format must be csv or json")
        return self


class DataAnalysisToolPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    evidence_id: str
    dataset: DatasetReference | None = None
    inspection: DatasetInspection | None = None
    analysis: AnalysisResult | None = None
    transform: DatasetTransformRecord | None = None
    artifact: DataAnalysisArtifact | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
