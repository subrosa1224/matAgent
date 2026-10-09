"""Planner data models (D2-M1)."""

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator

from materials_screening.llm.metadata import ProviderMetadata
from materials_screening.models import ScreeningRequest


class DraftStatus(StrEnum):
    """Status produced by the LLM draft."""

    EXTRACTED = "extracted"


class EnergyUnit(StrEnum):
    EV = "eV"
    MEV = "meV"
    UNSPECIFIED = "unspecified"


class HullUnit(StrEnum):
    EV_PER_ATOM = "eV/atom"
    MEV_PER_ATOM = "meV/atom"
    UNSPECIFIED = "unspecified"


class DensityUnit(StrEnum):
    G_PER_CM3 = "g/cm3"
    KG_PER_M3 = "kg/m3"
    UNSPECIFIED = "unspecified"


class PlannerStatus(StrEnum):
    READY = "ready"
    NEEDS_CLARIFICATION = "needs_clarification"
    INVALID = "invalid"
    UNSUPPORTED = "unsupported"


class EvidenceItem(BaseModel):
    """A short quote from the user query backing an extracted condition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    quote: str


class PlannerDraft(BaseModel):
    """Flat DTO produced by the LLM; never a trusted ScreeningRequest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: DraftStatus
    required_elements: list[str]
    excluded_elements: list[str]
    chemsys: str | None
    formula: str | None
    band_gap_min: float | None
    band_gap_max: float | None
    band_gap_unit: EnergyUnit
    hull_min: float | None
    hull_max: float | None
    hull_unit: HullUnit
    density_min: float | None
    density_max: float | None
    density_unit: DensityUnit
    crystal_system: str | None
    spacegroup_numbers: list[int]
    is_metal: bool | None
    is_stable: bool | None
    theoretical: bool | None
    target_band_gap: float | None
    target_band_gap_unit: EnergyUnit
    limit: int | None
    ambiguities: list[str]
    unsupported_requirements: list[str]
    conflicts: list[str]
    assumptions: list[str]
    clarification_question: str
    evidence: list[EvidenceItem]


class PlannerResult(BaseModel):
    """Deterministic outcome built by the Resolver from a PlannerDraft."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: PlannerStatus
    query: str
    request: ScreeningRequest | None = None
    clarification_question: str = ""
    invalid_reasons: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    ambiguities: tuple[str, ...] = ()
    unsupported_requirements: tuple[str, ...] = ()
    provider_metadata: ProviderMetadata | None = None

    @model_validator(mode="after")
    def validate_status_invariants(self) -> Self:
        if self.status is PlannerStatus.READY:
            if self.request is None:
                raise ValueError("READY results must include a request")
        elif self.request is not None:
            raise ValueError("non-READY results must not include a request")
        if (
            self.status is PlannerStatus.NEEDS_CLARIFICATION
            and not self.clarification_question.strip()
        ):
            raise ValueError(
                "NEEDS_CLARIFICATION results must include a clarification question"
            )
        if self.status is PlannerStatus.INVALID:
            if not self.invalid_reasons:
                raise ValueError("INVALID results must include invalid reasons")
        elif self.invalid_reasons:
            raise ValueError("non-INVALID results must not include invalid reasons")
        return self
