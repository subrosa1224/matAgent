"""Deterministic dataset inspection and data-quality profiling."""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
from pandas.api import types as pandas_types  # type: ignore[import-untyped]

from materials_screening.data_analysis.dataset_store import DatasetStore
from materials_screening.data_analysis.models import (
    ColumnProfile,
    DataQualityIssue,
    DatasetInspection,
    InferredColumnType,
    JsonScalar,
)


class DataAnalysisService:
    """Read-only DA-1 dataset inspection service."""

    def __init__(self, store: DatasetStore) -> None:
        self.store = store

    def inspect_dataset(
        self, dataset_id: str, *, offset: int = 0, limit: int = 20
    ) -> DatasetInspection:
        if offset < 0:
            raise ValueError("preview offset must be non-negative")
        if not 1 <= limit <= 100:
            raise ValueError("preview limit must be between 1 and 100")
        reference = self.store.get(dataset_id)
        frame = self.store.load_dataframe(dataset_id)
        if offset > len(frame.index):
            raise ValueError("preview offset exceeds dataset row count")

        profiles: list[ColumnProfile] = []
        issues: list[DataQualityIssue] = []
        for name in frame.columns:
            series = frame[name]
            column_name = str(name)
            missing_count = int(series.isna().sum())
            non_null = series.dropna()
            warnings: list[str] = []
            if missing_count:
                warnings.append("存在缺失值")
                issues.append(
                    DataQualityIssue(
                        code="missing_values",
                        severity="warning",
                        message=f"字段 {column_name} 存在 {missing_count} 个缺失值。",
                        column=column_name,
                        count=missing_count,
                    )
                )
            unique_count = int(non_null.nunique(dropna=True))
            if len(non_null.index) > 0 and unique_count <= 1:
                warnings.append("字段为常量或仅有一个有效取值")
                issues.append(
                    DataQualityIssue(
                        code="constant_column",
                        severity="warning",
                        message=f"字段 {column_name} 缺少可用于比较的变化。",
                        column=column_name,
                        count=unique_count,
                    )
                )
            mixed = _has_mixed_python_types(non_null)
            if mixed:
                warnings.append("字段包含混合数据类型")
                issues.append(
                    DataQualityIssue(
                        code="mixed_types",
                        severity="warning",
                        message=f"字段 {column_name} 包含混合数据类型。",
                        column=column_name,
                    )
                )
            inferred_type = _infer_column_type(series)
            if inferred_type == "numeric":
                outlier_count = _iqr_outlier_count(series)
                if outlier_count:
                    warnings.append(f"IQR 规则提示 {outlier_count} 个疑似异常值")
                    issues.append(
                        DataQualityIssue(
                            code="iqr_outlier_candidates",
                            severity="info",
                            message=(
                                f"字段 {column_name} 有 {outlier_count} 个值位于 "
                                "1.5×IQR 边界之外。"
                            ),
                            column=column_name,
                            count=outlier_count,
                        )
                    )
            samples = tuple(_json_scalar(value) for value in non_null.head(5))
            profiles.append(
                ColumnProfile(
                    name=column_name,
                    inferred_type=inferred_type,
                    non_null_count=len(non_null.index),
                    missing_count=missing_count,
                    unique_count=unique_count,
                    sample_values=samples,
                    warnings=tuple(warnings),
                )
            )

        duplicate_count = int(frame.duplicated().sum())
        if duplicate_count:
            issues.append(
                DataQualityIssue(
                    code="duplicate_rows",
                    severity="warning",
                    message=f"数据集存在 {duplicate_count} 条完全重复记录。",
                    count=duplicate_count,
                )
            )
        preview = tuple(
            {
                str(name): _json_scalar(value)
                for name, value in row.items()
            }
            for row in frame.iloc[offset : offset + limit].to_dict(orient="records")
        )
        return DatasetInspection(
            dataset=reference,
            columns=tuple(profiles),
            preview_offset=offset,
            preview_rows=preview,
            duplicate_row_count=duplicate_count,
            issues=tuple(issues),
        )


def _infer_column_type(series: pd.Series[Any]) -> InferredColumnType:
    if pandas_types.is_bool_dtype(series.dtype):
        return "boolean"
    if pandas_types.is_numeric_dtype(series.dtype):
        return "numeric"
    if pandas_types.is_datetime64_any_dtype(series.dtype):
        return "datetime"
    non_null = series.dropna()
    if not non_null.empty and _looks_like_iso_datetime(non_null):
        return "datetime"
    unique_count = int(non_null.nunique(dropna=True))
    if unique_count <= 50 and unique_count <= max(20, len(non_null.index) // 5):
        return "categorical"
    return "string"


def _looks_like_iso_datetime(series: pd.Series[Any]) -> bool:
    values = series.astype(str)
    if not bool(values.str.match(r"^\d{4}-\d{2}-\d{2}(?:[T ][^\s]+)?$").all()):
        return False
    parsed = pd.to_datetime(values, errors="coerce")
    return bool(parsed.notna().all())


def _has_mixed_python_types(series: pd.Series[Any]) -> bool:
    if not pandas_types.is_object_dtype(series.dtype) or series.empty:
        return False
    kinds = {
        "bool" if isinstance(value, bool) else type(value).__name__
        for value in series.head(1000)
    }
    return len(kinds) > 1


def _iqr_outlier_count(series: pd.Series[Any]) -> int:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    numeric = numeric[numeric.map(lambda value: math.isfinite(float(value)))]
    if len(numeric.index) < 4:
        return 0
    first = float(numeric.quantile(0.25))
    third = float(numeric.quantile(0.75))
    spread = third - first
    if spread <= 0:
        return 0
    lower = first - 1.5 * spread
    upper = third + 1.5 * spread
    return int(((numeric < lower) | (numeric > upper)).sum())


def _json_scalar(value: Any) -> JsonScalar:
    if value is None or bool(pd.isna(value)):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, datetime | date | pd.Timestamp):
        return value.isoformat()
    if isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return str(value)
