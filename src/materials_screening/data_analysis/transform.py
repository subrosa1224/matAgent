"""Validated, atomic transformations for immutable tabular datasets."""

from __future__ import annotations

import math
import uuid
from collections.abc import Mapping, Sequence, Set
from typing import Any

import pandas as pd  # type: ignore[import-untyped]

from materials_screening.data_analysis.dataset_store import DatasetStore
from materials_screening.data_analysis.models import (
    DatasetInspection,
    DatasetReference,
    DatasetTransformOperation,
    DatasetTransformRecord,
)
from materials_screening.data_analysis.service import DataAnalysisService


class DatasetTransformService:
    """Apply an ordered whitelist of transformations to an immutable source."""

    def __init__(self, store: DatasetStore) -> None:
        self.store = store

    def transform_dataset(
        self,
        dataset_id: str,
        *,
        operations: Sequence[DatasetTransformOperation],
        operation_id: str | None = None,
    ) -> tuple[DatasetReference, DatasetTransformRecord, DatasetInspection]:
        checked = tuple(operations)
        if not checked or len(checked) > 32:
            raise ValueError("provide between 1 and 32 transform operations")
        frame = self.store.load_dataframe(dataset_id).copy(deep=True)
        for operation in checked:
            frame = _apply_operation(frame, operation)
            if frame.empty:
                raise ValueError("transform operations produced an empty dataset")
            if not 1 <= len(frame.columns) <= 500:
                raise ValueError("transform result has an invalid column count")
            if len(frame.index) > 100_000:
                raise ValueError("transform result exceeds the row limit")
        safe_frame, escaped_count = _escape_formula_cells(frame)
        warnings = (
            (f"已安全转义 {escaped_count} 个可能触发电子表格公式的文本单元格。",)
            if escaped_count
            else ()
        )
        resolved_operation_id = operation_id or f"operation-{uuid.uuid4().hex}"
        reference, record = self.store.save_derived_dataset(
            safe_frame,
            source_dataset_id=dataset_id,
            operation_id=resolved_operation_id,
            operations=checked,
            warnings=warnings,
        )
        inspection = DataAnalysisService(self.store).inspect_dataset(
            reference.dataset_id
        )
        return reference, record, inspection


def _apply_operation(
    frame: pd.DataFrame, operation: DatasetTransformOperation
) -> pd.DataFrame:
    parameters = operation.parameters
    if operation.kind == "select_columns":
        _expect_keys(parameters, required={"columns"})
        columns = _column_list(parameters["columns"])
        _require_columns(frame, columns)
        return frame.loc[:, columns].copy()
    if operation.kind == "filter_rows":
        return _filter_rows(frame, parameters)
    if operation.kind == "drop_duplicates":
        _expect_keys(parameters, optional={"subset", "keep"})
        subset = (
            _column_list(parameters["subset"])
            if parameters.get("subset") is not None
            else None
        )
        if subset is not None:
            _require_columns(frame, subset)
        keep = parameters.get("keep", "first")
        if keep not in {"first", "last"}:
            raise ValueError("drop_duplicates keep must be first or last")
        return frame.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)
    if operation.kind == "convert_type":
        return _convert_type(frame, parameters)
    if operation.kind == "drop_missing":
        _expect_keys(parameters, optional={"subset", "how"})
        subset = (
            _column_list(parameters["subset"])
            if parameters.get("subset") is not None
            else None
        )
        if subset is not None:
            _require_columns(frame, subset)
        how = parameters.get("how", "any")
        if how not in {"any", "all"}:
            raise ValueError("drop_missing how must be any or all")
        return frame.dropna(subset=subset, how=how).reset_index(drop=True)
    if operation.kind == "fill_missing":
        return _fill_missing(frame, parameters)
    raise ValueError(f"unsupported transform operation: {operation.kind}")


def _filter_rows(frame: pd.DataFrame, parameters: Mapping[str, Any]) -> pd.DataFrame:
    operator = parameters.get("operator")
    if operator in {"in", "not_in"}:
        _expect_keys(parameters, required={"column", "operator", "values"})
    else:
        _expect_keys(parameters, required={"column", "operator", "value"})
    column = _column_name(parameters["column"])
    _require_columns(frame, [column])
    series = frame[column]
    if operator == "eq":
        mask = series == _json_scalar_parameter(parameters["value"])
    elif operator == "ne":
        mask = series != _json_scalar_parameter(parameters["value"])
    elif operator in {"gt", "ge", "lt", "le"}:
        value = _json_scalar_parameter(parameters["value"])
        try:
            mask = {
                "gt": series > value,
                "ge": series >= value,
                "lt": series < value,
                "le": series <= value,
            }[operator]
        except TypeError as exc:
            raise ValueError(
                "filter comparison is incompatible with column type"
            ) from exc
    elif operator in {"in", "not_in"}:
        values = parameters["values"]
        if not isinstance(values, list) or not 1 <= len(values) <= 1000:
            raise ValueError("filter values must contain between 1 and 1000 items")
        values = [_json_scalar_parameter(value) for value in values]
        mask = series.isin(values)
        if operator == "not_in":
            mask = ~mask
    else:
        raise ValueError("unsupported filter operator")
    return frame.loc[mask.fillna(False)].reset_index(drop=True)


def _convert_type(frame: pd.DataFrame, parameters: Mapping[str, Any]) -> pd.DataFrame:
    _expect_keys(parameters, required={"column", "dtype"})
    column = _column_name(parameters["column"])
    _require_columns(frame, [column])
    dtype = parameters["dtype"]
    result = frame.copy()
    try:
        if dtype == "numeric":
            result[column] = pd.to_numeric(result[column], errors="raise")
        elif dtype == "string":
            result[column] = result[column].astype("string")
        elif dtype == "boolean":
            result[column] = result[column].map(_to_boolean).astype("boolean")
        elif dtype == "datetime":
            converted = pd.to_datetime(result[column], errors="raise")
            result[column] = converted.dt.strftime("%Y-%m-%dT%H:%M:%S")
        else:
            raise ValueError("dtype must be numeric, string, boolean or datetime")
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"column {column!r} cannot be converted to {dtype}") from exc
    return result


def _fill_missing(frame: pd.DataFrame, parameters: Mapping[str, Any]) -> pd.DataFrame:
    strategy = parameters.get("strategy")
    required = {"column", "strategy", "value"} if strategy == "constant" else {
        "column",
        "strategy",
    }
    _expect_keys(parameters, required=required, optional={"group_by"})
    column = _column_name(parameters["column"])
    _require_columns(frame, [column])
    group_by = parameters.get("group_by")
    if group_by is not None:
        group_by = _column_name(group_by)
        _require_columns(frame, [group_by])
    result = frame.copy()
    if strategy == "constant":
        value = parameters["value"]
        value = _json_scalar_parameter(value)
        result[column] = result[column].fillna(value)
    elif strategy in {"mean", "median"}:
        numeric = pd.to_numeric(result[column], errors="coerce")
        if int(numeric.notna().sum()) != int(result[column].notna().sum()):
            raise ValueError("mean/median fill requires a numeric column")
        if group_by is None:
            fill_value = numeric.mean() if strategy == "mean" else numeric.median()
            if pd.isna(fill_value):
                raise ValueError("cannot fill a column with no numeric values")
            result[column] = numeric.fillna(fill_value)
        else:
            grouped = numeric.groupby(result[group_by], dropna=False)
            fill_values = grouped.transform(strategy)
            result[column] = numeric.fillna(fill_values)
            if bool(result[column].isna().any()):
                raise ValueError(
                    "at least one group has no value available for filling"
                )
    else:
        raise ValueError("fill strategy must be constant, mean or median")
    return result


def _to_boolean(value: Any) -> bool | None:
    if value is None or bool(pd.isna(value)):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float) and value in {0, 1}:
        return bool(value)
    normalized = str(value).strip().casefold()
    if normalized in {"true", "yes", "y", "1"}:
        return True
    if normalized in {"false", "no", "n", "0"}:
        return False
    raise ValueError(f"unsupported boolean value: {value!r}")


def _escape_formula_cells(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    result = frame.copy()
    escaped = 0
    for column in result.select_dtypes(include=["object", "string"]).columns:
        def escape(value: Any) -> Any:
            nonlocal escaped
            if isinstance(value, str) and value.startswith(("=", "+", "-", "@")):
                escaped += 1
                return "'" + value
            return value

        result[column] = result[column].map(escape)
    return result, escaped


def _expect_keys(
    parameters: Mapping[str, Any],
    *,
    required: Set[str] = frozenset(),
    optional: Set[str] = frozenset(),
) -> None:
    keys = set(parameters)
    if missing := required - keys:
        raise ValueError(f"missing transform parameters: {sorted(missing)}")
    if extra := keys - required - optional:
        raise ValueError(f"unexpected transform parameters: {sorted(extra)}")


def _column_list(value: Any) -> list[str]:
    if not isinstance(value, list) or not value or len(value) > 500:
        raise ValueError("columns must be a non-empty list with at most 500 items")
    columns = [_column_name(item) for item in value]
    if len(columns) != len(set(columns)):
        raise ValueError("columns must be unique")
    return columns


def _column_name(value: Any) -> str:
    if not isinstance(value, str) or not value or len(value) > 256:
        raise ValueError("column name must be a bounded non-blank string")
    return value


def _json_scalar_parameter(value: Any) -> Any:
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError("transform value must be a finite JSON scalar")


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"unknown dataset columns: {missing}")
