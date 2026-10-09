"""Validated contracts for database queries and analyses."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

NumericField = Literal[
    "band_gap_ev",
    "density_g_cm3",
    "formation_energy_ev_atom",
    "energy_above_hull_ev_atom",
]
FilterField = Literal[
    "band_gap_ev",
    "density_g_cm3",
    "formation_energy_ev_atom",
    "energy_above_hull_ev_atom",
    "material_id",
    "formula_pretty",
    "chemsys",
    "is_stable",
    "is_metal",
    "theoretical",
    "crystal_system",
    "spacegroup_number",
]
OutputField = Literal[
    "band_gap_ev",
    "density_g_cm3",
    "formation_energy_ev_atom",
    "energy_above_hull_ev_atom",
    "material_id",
    "formula_pretty",
    "chemsys",
    "is_stable",
    "is_metal",
    "theoretical",
    "crystal_system",
    "spacegroup_number",
    "elements",
    "spacegroup_symbol",
    "is_gap_direct",
    "deprecated",
    "source",
]
SortField = Literal[
    "band_gap_ev",
    "density_g_cm3",
    "formation_energy_ev_atom",
    "energy_above_hull_ev_atom",
    "formula_pretty",
    "material_id",
]

NUMERIC_FIELDS = frozenset(
    {
        "band_gap_ev",
        "density_g_cm3",
        "formation_energy_ev_atom",
        "energy_above_hull_ev_atom",
    }
)
FILTER_FIELDS = NUMERIC_FIELDS | frozenset(
    {
        "material_id",
        "formula_pretty",
        "chemsys",
        "is_stable",
        "is_metal",
        "theoretical",
        "crystal_system",
        "spacegroup_number",
    }
)
OUTPUT_FIELDS = FILTER_FIELDS | frozenset(
    {
        "elements",
        "spacegroup_symbol",
        "is_gap_direct",
        "deprecated",
        "source",
    }
)


class FilterOperator(StrEnum):
    EQ = "eq"
    GTE = "gte"
    LTE = "lte"
    GT = "gt"
    LT = "lt"


class PropertyFilter(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    field: FilterField
    operator: FilterOperator
    value: str | float | int | bool

class SortRule(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    field: SortField
    direction: Literal["asc", "desc"] = "asc"

class SearchMaterialsInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    required_elements: tuple[str, ...] = ()
    excluded_elements: tuple[str, ...] = ()
    chemsys: str | None = None
    formula: str | None = None
    material_ids: tuple[str, ...] = ()
    num_elements: int | None = Field(default=None, ge=1, le=10)
    filters: tuple[PropertyFilter, ...] = ()
    sort: tuple[SortRule, ...] = ()
    limit: int = Field(default=20, ge=1, le=1000)
    fields: tuple[OutputField, ...] = ("material_id", "formula_pretty")


class QueryResultReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    query_id: str
    source: Literal["materials_project"]
    matched_count: int
    returned_count: int
    fields: tuple[str, ...]
    materials: tuple[dict[str, Any], ...]
    warnings: tuple[str, ...] = ()
    evidence_id: str | None = None
    created_at: datetime


class GetMaterialDetailsInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    material_ids: tuple[str, ...] = Field(min_length=1, max_length=20)
    fields: tuple[OutputField, ...] = ("material_id", "formula_pretty")


class GetQueryResultInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    query_id: str
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=20, ge=1, le=100)
    fields: tuple[OutputField, ...] = ("material_id", "formula_pretty")


class CompareMaterialsInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    material_ids: tuple[str, ...] = Field(default=(), max_length=20)
    query_id: str | None = None
    properties: tuple[NumericField, ...] = Field(min_length=1)


class DescribeMaterialsInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    query_id: str
    properties: tuple[NumericField, ...] = Field(min_length=1)
    include_correlation: bool = False


class DetectOutliersInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    query_id: str
    properties: tuple[NumericField, ...] = Field(min_length=1)
    method: Literal["zscore", "iqr", "mahalanobis", "isolation_forest"] = "iqr"
    threshold: float = Field(default=1.5, gt=0)


class ExportMaterialsInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    query_id: str | None = None
    analysis_id: str | None = None
    format: Literal["csv", "json", "markdown"] = "csv"
    scope: Literal["all", "top_k", "page"] = "all"
    fields: tuple[OutputField, ...] = ()

    @model_validator(mode="after")
    def exactly_one_reference(self) -> Self:
        if (self.query_id is None) == (self.analysis_id is None):
            raise ValueError("provide exactly one of query_id or analysis_id")
        return self


class ToolPayload(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)
    evidence_id: str | None = None
