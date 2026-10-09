"""Domain models for the screening core (stage 1, M2)."""

from datetime import datetime
from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from materials_screening.chemistry import normalize_chemsys, normalize_element_list


class FloatRange(BaseModel):
    """Inclusive numeric range that requires at least one bound."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    min: float | None = Field(default=None, allow_inf_nan=False)
    max: float | None = Field(default=None, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_order(self) -> Self:
        if self.min is None and self.max is None:
            raise ValueError("At least one bound must be provided")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("min cannot be greater than max")
        return self


class CrystalSystem(StrEnum):
    """Canonical crystal system names used in screening requests."""

    TRICLINIC = "Triclinic"
    MONOCLINIC = "Monoclinic"
    ORTHORHOMBIC = "Orthorhombic"
    TETRAGONAL = "Tetragonal"
    TRIGONAL = "Trigonal"
    HEXAGONAL = "Hexagonal"
    CUBIC = "Cubic"


class ScreeningRequest(BaseModel):
    """Structured screening request with validated chemistry constraints."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    required_elements: tuple[str, ...] = ()
    excluded_elements: tuple[str, ...] = ()

    chemsys: str | None = None
    formula: str | None = None
    material_ids: tuple[str, ...] = ()
    num_elements: int | None = Field(default=None, ge=1, le=10)
    num_elements: int | None = Field(default=None, ge=1, le=10)

    band_gap_ev: FloatRange | None = None
    energy_above_hull_ev_atom: FloatRange | None = None
    density_g_cm3: FloatRange | None = None

    crystal_system: CrystalSystem | None = None
    spacegroup_numbers: tuple[int, ...] = ()

    is_metal: bool | None = False
    is_stable: bool | None = None
    theoretical: bool | None = None

    target_band_gap_ev: float | None = Field(default=None, allow_inf_nan=False)
    limit: int = Field(default=10, ge=1, le=100)

    @field_validator("required_elements", "excluded_elements")
    @classmethod
    def normalize_elements(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return normalize_element_list(values)

    @field_validator("chemsys")
    @classmethod
    def normalize_chemsys_field(cls, value: str | None) -> str | None:
        return normalize_chemsys(value)

    @field_validator("formula")
    @classmethod
    def empty_string_to_none(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            return None
        return value

    @field_validator("band_gap_ev", "energy_above_hull_ev_atom")
    @classmethod
    def non_negative_range(cls, value: FloatRange | None) -> FloatRange | None:
        if value is None:
            return None
        if value.min is not None and value.min < 0:
            raise ValueError("Range bounds must be non-negative")
        if value.max is not None and value.max < 0:
            raise ValueError("Range bounds must be non-negative")
        return value

    @field_validator("spacegroup_numbers")
    @classmethod
    def normalize_spacegroup_numbers(cls, values: tuple[int, ...]) -> tuple[int, ...]:
        for number in values:
            if number < 1 or number > 230:
                raise ValueError("Spacegroup number must be between 1 and 230")
        return tuple(sorted(set(values)))

    @model_validator(mode="after")
    def validate_element_conflicts(self) -> Self:
        overlap = set(self.required_elements) & set(self.excluded_elements)
        if overlap:
            joined = ", ".join(sorted(overlap))
            raise ValueError(
                f"Elements must not be both required and excluded: {joined}"
            )
        return self

    @model_validator(mode="after")
    def validate_target_band_gap(self) -> Self:
        target = self.target_band_gap_ev
        if target is None:
            return self
        if target < 0:
            raise ValueError("target_band_gap_ev must be non-negative")
        band_range = self.band_gap_ev
        if band_range is None:
            return self
        if band_range.min is not None and target < band_range.min:
            raise ValueError("target_band_gap_ev must be within band_gap_ev")
        if band_range.max is not None and target > band_range.max:
            raise ValueError("target_band_gap_ev must be within band_gap_ev")
        return self


class PropertyValueType(StrEnum):
    """Classification of how a material property was obtained."""

    DFT_CALCULATED = "dft_calculated"
    EXPERIMENTAL = "experimental"
    ML_PREDICTED = "ml_predicted"
    DERIVED = "derived"
    UNKNOWN = "unknown"


class PropertyProvenance(BaseModel):
    """Provenance record for a single material property."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    property_name: str
    source: str
    source_material_id: str
    value_type: PropertyValueType
    database_version: str | None
    retrieved_at: datetime
    method: str | None = None


class SymmetryInfo(BaseModel):
    """Space group information for a material."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    crystal_system: CrystalSystem | None = None
    symbol: str | None = None
    number: int | None = Field(default=None, ge=1, le=230)


class MaterialRecord(BaseModel):
    """A screened material with validated physical properties."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: str
    material_id: str
    formula_pretty: str
    elements: tuple[str, ...]
    chemsys: str | None = None

    band_gap_ev: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    energy_above_hull_ev_atom: float | None = Field(
        default=None, ge=0, allow_inf_nan=False
    )
    formation_energy_ev_atom: float | None = Field(default=None, allow_inf_nan=False)
    density_g_cm3: float | None = Field(default=None, gt=0, allow_inf_nan=False)

    is_metal: bool | None = None
    is_gap_direct: bool | None = None
    is_stable: bool | None = None
    theoretical: bool | None = None
    deprecated: bool | None = None

    symmetry: SymmetryInfo | None = None

    structure_dict: dict[str, Any] | None = None
    structure_hash: str | None = None

    provenance: tuple[PropertyProvenance, ...] = ()


class Rejection(BaseModel):
    """A rejected candidate and the reasons for rejection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    material_id: str
    reasons: tuple[str, ...]


class FilterStep(BaseModel):
    """Counts for one hard-filter step."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    before_count: int
    after_count: int
    rejection_count: int
    reason_counts: dict[str, int]


class FilterTrace(BaseModel):
    """Trace of all hard-filter steps and per-material rejections."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    steps: tuple[FilterStep, ...]
    rejections: tuple[Rejection, ...]


class ScoreBreakdown(BaseModel):
    """Weighted and unweighted components of a ranking score."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    stability: float
    band_gap_match: float
    completeness: float
    direct_gap: float

    weighted_stability: float
    weighted_band_gap_match: float
    weighted_completeness: float
    weighted_direct_gap: float


class RankedMaterial(BaseModel):
    """A material with its final rank and score breakdown."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    record: MaterialRecord
    rank: int = Field(ge=1)
    total_score: float = Field(ge=0, le=1)
    score_breakdown: ScoreBreakdown


class ValidationReport(BaseModel):
    """Result of re-validating the final ranked materials."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    passed: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    checked_material_ids: tuple[str, ...] = ()


class RunMetadata(BaseModel):
    """Metadata describing one screening run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    started_at: datetime
    finished_at: datetime
    source: str
    database_version: str | None
    mp_api_version: str | None
    pymatgen_version: str | None
    application_version: str
    query_fingerprint: str


class ScreeningResult(BaseModel):
    """Complete result of one screening run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    request: ScreeningRequest
    metadata: RunMetadata
    retrieved_count: int
    passed_filter_count: int
    ranked_materials: tuple[RankedMaterial, ...]
    filter_trace: FilterTrace
    validation: ValidationReport
