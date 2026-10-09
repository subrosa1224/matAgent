"""Input / output models for the Outlier Detection sub-agent."""

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ---------------------------------------------------------------------------
# Whitelisted properties
# ---------------------------------------------------------------------------

ALLOWED_PROPERTIES: frozenset[str] = frozenset({
    "band_gap_ev",
    "formation_energy_ev_atom",
    "energy_above_hull_ev_atom",
    "density_g_cm3",
})

ALLOWED_UNIVARIATE_METHODS: frozenset[str] = frozenset({"zscore", "iqr"})
ALLOWED_MULTIVARIATE_METHODS: frozenset[str] = frozenset(
    {"mahalanobis", "isolation_forest"}
)

DEFAULT_ZSCORE_THRESHOLD = 2.0
DEFAULT_IQR_THRESHOLD = 1.5


# ---------------------------------------------------------------------------
# Material set reference
# ---------------------------------------------------------------------------


class MaterialSetReference(BaseModel):
    """User-provided reference to a set of materials for outlier analysis.

    Exactly one field must be provided. This is a lookup key for a
    collection of materials — not the material data itself.

    Outlier detection always operates on a population, so all three
    modes resolve to a collection of MaterialRecords.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    workflow_thread_ids: tuple[str, ...] | None = None
    # e.g. ("run-123", "run-456") from screening workflow results
    material_formulas: tuple[str, ...] | None = None  # e.g. ("TiO2", "BaTiO3")
    material_ids: tuple[str, ...] | None = None       # e.g. ("mp-149", "mp-2657")
    data_file: str | None = None                      # path to CSV / JSON file

    @model_validator(mode="after")
    def validate_exactly_one(self) -> "MaterialSetReference":
        provided = [
            self.workflow_thread_ids is not None,
            self.material_formulas is not None,
            self.material_ids is not None,
            self.data_file is not None,
        ]
        if sum(provided) != 1:
            raise ValueError(
                "workflow_thread_ids、material_formulas、data_file 必须且只能提供一个"
            )
        return self


class RunOutlierDetectionInput(BaseModel):
    """Natural-language request accepted by the public outlier tool."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str = Field(min_length=1, max_length=8192)


# ---------------------------------------------------------------------------
# Tool 1: detect_property_outliers
# ---------------------------------------------------------------------------


class DetectPropertyOutliersInput(BaseModel):
    """Input for single-property outlier detection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: MaterialSetReference
    property: str                  # whitelisted property name
    method: str = "zscore"         # "zscore" | "iqr"
    threshold: float | None = None  # default: zscore→2.0, iqr→1.5


class PropertyOutlierRecord(BaseModel):
    """Single material result for single-property outlier detection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    material_id: str | None       # None when input has no mp-id (e.g. user CSV)
    material_label: str           # formula, mp-id, or user-provided identifier
    property: str                 # echo property name
    value: float
    z_score: float | None
    is_outlier: bool
    direction: str                # "high" | "low"


class DistributionStats(BaseModel):
    """Single-property distribution statistics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    count: int
    mean: float
    median: float
    std: float
    q1: float
    q3: float
    iqr: float
    min_value: float
    max_value: float


class PropertyOutlierReport(BaseModel):
    """Output of detect_property_outliers."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    property_name: str
    method: str
    threshold: float
    distribution: DistributionStats
    records: tuple[PropertyOutlierRecord, ...]
    warnings: tuple[str, ...] = ()
    evidence_id: str = ""


# ---------------------------------------------------------------------------
# Tool 2: detect_multivariate_outliers
# ---------------------------------------------------------------------------


class DetectMultivariateOutliersInput(BaseModel):
    """Input for multivariate outlier detection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: MaterialSetReference
    properties: tuple[str, ...]        # 2–4 whitelisted properties
    method: str = "mahalanobis"        # "mahalanobis" | "isolation_forest"

    @model_validator(mode="after")
    def validate_properties_count(self) -> "DetectMultivariateOutliersInput":
        if len(self.properties) < 2:
            raise ValueError("multivariate detection requires at least 2 properties")
        if len(self.properties) > 4:
            raise ValueError("multivariate detection supports at most 4 properties")
        return self


class MultivariateOutlierRecord(BaseModel):
    """Single material result for multivariate outlier detection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    material_id: str | None            # None when input has no mp-id
    material_label: str                # formula, mp-id, or user-provided identifier
    anomaly_score: float               # Mahalanobis distance or normalised score
    is_outlier: bool = False           # True when score exceeds statistical threshold
    abnormal_properties: tuple[str, ...]  # properties contributing most to the anomaly
    outlier_reason: str                # e.g. "高带隙+低形成能+密度异常"


class MultivariateOutlierReport(BaseModel):
    """Output of detect_multivariate_outliers."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    properties: tuple[str, ...]
    method: str
    threshold: float | None = None
    records: tuple[MultivariateOutlierRecord, ...]
    warnings: tuple[str, ...] = ()
    evidence_id: str = ""
