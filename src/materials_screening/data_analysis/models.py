"""Strict domain contracts for deterministic tabular data analysis."""

from __future__ import annotations

import math
import re
from typing import Any, Literal, Self, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DatasetFormat: TypeAlias = Literal["csv", "json", "xlsx"]
InferredColumnType: TypeAlias = Literal[
    "numeric", "boolean", "string", "categorical", "datetime"
]
AnalysisType: TypeAlias = Literal[
    "quality", "descriptive", "correlation", "statistical_test", "regression"
]
TransformKind: TypeAlias = Literal[
    "select_columns",
    "filter_rows",
    "drop_duplicates",
    "convert_type",
    "drop_missing",
    "fill_missing",
]
DataAnalysisArtifactType: TypeAlias = Literal[
    "dataset_file", "analysis_result", "analysis_plot", "analysis_report"
]
ArtifactFileExtension: TypeAlias = Literal["csv", "json", "md", "png", "docx"]
QualitySeverity: TypeAlias = Literal["info", "warning", "error"]
JsonScalar: TypeAlias = str | int | float | bool | None

_DATASET_ID = r"^dataset-[A-Za-z0-9][A-Za-z0-9._:-]{0,247}$"
_ANALYSIS_ID = r"^analysis-[A-Za-z0-9][A-Za-z0-9._:-]{0,246}$"
_OPERATION_ID = r"^operation-[A-Za-z0-9][A-Za-z0-9._:-]{0,245}$"
_ARTIFACT_ID = r"^artifact-[A-Za-z0-9][A-Za-z0-9._:-]{0,246}$"
_EVIDENCE_ID = r"^evidence-[A-Za-z0-9][A-Za-z0-9._:-]{0,246}$"
_SHA256 = r"^sha256:[0-9a-f]{64}$"


class DatasetReference(BaseModel):
    """Stable public metadata for one immutable tabular dataset version."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: str = Field(pattern=_DATASET_ID)
    source_artifact_id: str = Field(pattern=_ARTIFACT_ID)
    display_name: str = Field(min_length=1, max_length=512)
    format: DatasetFormat
    row_count: int = Field(ge=0, le=100_000)
    column_count: int = Field(ge=1, le=500)
    schema_fingerprint: str = Field(pattern=_SHA256)
    content_fingerprint: str = Field(pattern=_SHA256)
    parent_dataset_id: str | None = Field(default=None, pattern=_DATASET_ID)
    created_by_operation_id: str | None = Field(
        default=None, pattern=_OPERATION_ID
    )

    @model_validator(mode="after")
    def validate_derivation_link(self) -> Self:
        has_parent = self.parent_dataset_id is not None
        has_operation = self.created_by_operation_id is not None
        if has_parent != has_operation:
            raise ValueError(
                "derived datasets require both parent_dataset_id and "
                "created_by_operation_id"
            )
        if self.parent_dataset_id == self.dataset_id:
            raise ValueError("a dataset cannot be its own parent")
        return self


class ColumnProfile(BaseModel):
    """Bounded, JSON-safe profile for one dataset column."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, max_length=256)
    inferred_type: InferredColumnType
    non_null_count: int = Field(ge=0, le=100_000)
    missing_count: int = Field(ge=0, le=100_000)
    unique_count: int = Field(ge=0, le=100_000)
    sample_values: tuple[JsonScalar, ...] = Field(default=(), max_length=10)
    warnings: tuple[str, ...] = Field(default=(), max_length=32)

    @field_validator("sample_values")
    @classmethod
    def validate_finite_samples(
        cls, values: tuple[JsonScalar, ...]
    ) -> tuple[JsonScalar, ...]:
        for value in values:
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("sample_values must not contain NaN or infinity")
        return values

    @field_validator("warnings")
    @classmethod
    def validate_warnings(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not value.strip() or len(value) > 1000 for value in values):
            raise ValueError("warnings must be non-blank and at most 1000 characters")
        return values

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        if self.unique_count > self.non_null_count:
            raise ValueError("unique_count must not exceed non_null_count")
        return self


class DataQualityIssue(BaseModel):
    """One bounded, user-safe issue discovered during dataset inspection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str = Field(min_length=1, max_length=128, pattern=r"^[a-z][a-z0-9_]*$")
    severity: QualitySeverity
    message: str = Field(min_length=1, max_length=1000)
    column: str | None = Field(default=None, min_length=1, max_length=256)
    count: int | None = Field(default=None, ge=0, le=100_000)


class DatasetInspection(BaseModel):
    """Bounded quality profile and page preview for one dataset."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset: DatasetReference
    columns: tuple[ColumnProfile, ...] = Field(max_length=500)
    preview_offset: int = Field(ge=0, le=100_000)
    preview_rows: tuple[dict[str, JsonScalar], ...] = Field(max_length=100)
    duplicate_row_count: int = Field(ge=0, le=100_000)
    issues: tuple[DataQualityIssue, ...] = Field(default=(), max_length=1000)

    @field_validator("preview_rows")
    @classmethod
    def validate_preview_rows(
        cls, rows: tuple[dict[str, JsonScalar], ...]
    ) -> tuple[dict[str, JsonScalar], ...]:
        for row in rows:
            if len(row) > 500:
                raise ValueError("preview rows may contain at most 500 columns")
            _validate_json_value(row)
        return rows

    @model_validator(mode="after")
    def validate_dataset_shape(self) -> Self:
        if len(self.columns) != self.dataset.column_count:
            raise ValueError("column profiles must match dataset.column_count")
        names = [profile.name for profile in self.columns]
        if len(names) != len(set(names)):
            raise ValueError("column profile names must be unique")
        for profile in self.columns:
            if profile.non_null_count + profile.missing_count != self.dataset.row_count:
                raise ValueError("column profile counts must match dataset.row_count")
        if self.preview_offset > self.dataset.row_count:
            raise ValueError("preview_offset exceeds dataset.row_count")
        if self.preview_offset + len(self.preview_rows) > self.dataset.row_count:
            raise ValueError("preview rows exceed dataset.row_count")
        expected_columns = set(names)
        if any(set(row) != expected_columns for row in self.preview_rows):
            raise ValueError("preview rows must match profiled columns")
        if any(
            issue.column is not None and issue.column not in expected_columns
            for issue in self.issues
        ):
            raise ValueError("quality issue references an unknown column")
        return self


class AnalysisResult(BaseModel):
    """Persistable, source-linked result of one deterministic analysis."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    analysis_id: str = Field(pattern=_ANALYSIS_ID)
    dataset_id: str = Field(pattern=_DATASET_ID)
    analysis_type: AnalysisType
    method: str = Field(min_length=1, max_length=128, pattern=r"^[a-z][a-z0-9_]*$")
    parameters: dict[str, Any] = Field(default_factory=dict)
    summary: dict[str, Any] = Field(default_factory=dict)
    warnings: tuple[str, ...] = Field(default=(), max_length=64)
    evidence_id: str = Field(pattern=_EVIDENCE_ID)

    @field_validator("parameters", "summary")
    @classmethod
    def validate_json_object(cls, value: dict[str, Any]) -> dict[str, Any]:
        _validate_json_value(value)
        return value

    @field_validator("warnings")
    @classmethod
    def validate_warnings(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not value.strip() or len(value) > 1000 for value in values):
            raise ValueError("warnings must be non-blank and at most 1000 characters")
        return values


class DataAnalysisArtifact(BaseModel):
    """Public metadata for one privately stored analysis artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_id: str = Field(pattern=_ARTIFACT_ID)
    artifact_type: DataAnalysisArtifactType
    dataset_id: str = Field(pattern=_DATASET_ID)
    analysis_ids: tuple[str, ...] = Field(default=(), max_length=100)
    display_name: str = Field(min_length=1, max_length=512)
    file_extension: ArtifactFileExtension
    media_type: str = Field(min_length=1, max_length=128)
    size_bytes: int = Field(ge=1, le=100 * 1024 * 1024)
    content_fingerprint: str = Field(pattern=_SHA256)

    @model_validator(mode="after")
    def validate_analysis_ids(self) -> Self:
        if len(self.analysis_ids) != len(set(self.analysis_ids)):
            raise ValueError("analysis_ids must be unique")
        if any(not re.fullmatch(_ANALYSIS_ID, value) for value in self.analysis_ids):
            raise ValueError("analysis_ids contain an invalid id")
        return self


class DatasetTransformOperation(BaseModel):
    """One validated operation in an ordered dataset transformation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: TransformKind
    parameters: dict[str, Any]

    @field_validator("parameters")
    @classmethod
    def validate_parameters(cls, value: dict[str, Any]) -> dict[str, Any]:
        if len(value) > 32:
            raise ValueError("operation parameters may contain at most 32 keys")
        _validate_json_value(value)
        return value


class DatasetTransformRecord(BaseModel):
    """Audit record linking an immutable source and derived dataset."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    operation_id: str = Field(pattern=_OPERATION_ID)
    source_dataset_id: str = Field(pattern=_DATASET_ID)
    result_dataset_id: str = Field(pattern=_DATASET_ID)
    operations: tuple[DatasetTransformOperation, ...] = Field(
        min_length=1, max_length=32
    )
    rows_before: int = Field(ge=0, le=100_000)
    rows_after: int = Field(ge=0, le=100_000)
    warnings: tuple[str, ...] = Field(default=(), max_length=64)

    @model_validator(mode="after")
    def validate_distinct_datasets(self) -> Self:
        if self.source_dataset_id == self.result_dataset_id:
            raise ValueError("transform result must use a new dataset_id")
        return self

    @field_validator("warnings")
    @classmethod
    def validate_warnings(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not value.strip() or len(value) > 1000 for value in values):
            raise ValueError("warnings must be non-blank and at most 1000 characters")
        return values


def _validate_json_value(value: Any, *, depth: int = 0) -> None:
    """Reject non-JSON, non-finite and excessively nested contract payloads."""

    if depth > 8:
        raise ValueError("JSON payload nesting exceeds 8 levels")
    if value is None or isinstance(value, str | bool | int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON payload must not contain NaN or infinity")
        return
    if isinstance(value, list | tuple):
        if len(value) > 1000:
            raise ValueError("JSON array exceeds 1000 items")
        for item in value:
            _validate_json_value(item, depth=depth + 1)
        return
    if isinstance(value, dict):
        if len(value) > 1000:
            raise ValueError("JSON object exceeds 1000 keys")
        for key, item in value.items():
            if not isinstance(key, str) or not re.fullmatch(r"[^\x00]{1,256}", key):
                raise ValueError("JSON object keys must be bounded strings")
            _validate_json_value(item, depth=depth + 1)
        return
    raise ValueError(f"unsupported JSON value type: {type(value).__name__}")
