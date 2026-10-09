"""Validated contracts for research-grade materials screening projects."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from materials_screening.chemistry import normalize_chemsys, normalize_element_list


class CriterionRole(StrEnum):
    HARD_FILTER = "hard_filter"
    SOFT_RANK = "soft_rank"
    REPORT_ONLY = "report_only"


class CriterionOperator(StrEnum):
    EQ = "eq"
    GTE = "gte"
    LTE = "lte"
    BETWEEN = "between"
    TARGET = "target"
    PREFER_MIN = "prefer_min"
    PREFER_MAX = "prefer_max"


class MissingValuePolicy(StrEnum):
    EXCLUDE = "exclude"
    KEEP_WITH_WARNING = "keep_with_warning"
    FAIL_PROJECT = "fail_project"
    NOT_APPLICABLE = "not_applicable"


class SourceRole(StrEnum):
    DATABASE_CALCULATED = "database_calculated"
    DATABASE_METADATA = "database_metadata"
    DATABASE_REPORTED_EXPERIMENTAL = "database_reported_experimental"
    LITERATURE_EXPERIMENTAL = "literature_experimental"
    USER_DATA = "user_data"


class ProjectStatus(StrEnum):
    DRAFT = "draft"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    CONFIRMED = "confirmed"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MaterialScope(BaseModel):
    """Explicit crystalline-inorganic candidate scope."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    domain: str = Field(
        default="crystalline_inorganic",
        pattern="^crystalline_inorganic$",
    )
    required_elements: tuple[str, ...] = Field(default=(), max_length=30)
    excluded_elements: tuple[str, ...] = Field(default=(), max_length=30)
    chemsys: str | None = Field(default=None, max_length=200)
    formulas: tuple[str, ...] = Field(default=(), max_length=100)
    material_ids: tuple[str, ...] = Field(default=(), max_length=500)

    @field_validator("required_elements", "excluded_elements", mode="before")
    @classmethod
    def normalize_elements(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return normalize_element_list(value)

    @field_validator("chemsys", mode="before")
    @classmethod
    def normalize_chemical_system(cls, value: str | None) -> str | None:
        return normalize_chemsys(value)

    @field_validator("formulas", "material_ids", mode="before")
    @classmethod
    def normalize_identifiers(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(dict.fromkeys(item.strip() for item in value))
        if any(not item for item in cleaned):
            raise ValueError("scope identifiers must not be blank")
        return cleaned

    @model_validator(mode="after")
    def elements_do_not_conflict(self) -> Self:
        required = {element.casefold() for element in self.required_elements}
        excluded = {element.casefold() for element in self.excluded_elements}
        overlap = required & excluded
        if overlap:
            raise ValueError(
                "required_elements and excluded_elements overlap: "
                + ", ".join(sorted(overlap))
            )
        return self


class ProjectResourceLimits(BaseModel):
    """Hard upper bounds that keep a screening project finite."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_snapshot_limit: int = Field(default=500, ge=1, le=500)
    shortlist_limit: int = Field(default=10, ge=1, le=100)
    quick_literature_limit: int = Field(default=20, ge=0, le=20)
    deep_literature_limit: int = Field(default=5, ge=0, le=5)
    supplemental_search_limit: int = Field(default=1, ge=0, le=1)

    @model_validator(mode="after")
    def deep_review_is_subset(self) -> Self:
        if self.shortlist_limit > self.candidate_snapshot_limit:
            raise ValueError(
                "shortlist_limit cannot exceed candidate_snapshot_limit"
            )
        if self.deep_literature_limit > self.quick_literature_limit:
            raise ValueError(
                "deep_literature_limit cannot exceed quick_literature_limit"
            )
        return self


class ScreeningCriterion(BaseModel):
    """One explicit constraint, ranking preference, or reporting request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    property_id: str = Field(min_length=1, max_length=128)
    role: CriterionRole
    operator: CriterionOperator
    values: tuple[str | float | int | bool, ...]
    unit: str | None = Field(default=None, max_length=64)
    missing_policy: MissingValuePolicy
    weight: float | None = Field(default=None, gt=0, le=100, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_value_count(self) -> Self:
        expected = 2 if self.operator is CriterionOperator.BETWEEN else 1
        if len(self.values) != expected:
            raise ValueError(
                f"operator {self.operator.value!r} requires {expected} value(s)"
            )
        if self.role is CriterionRole.SOFT_RANK and self.weight is None:
            raise ValueError("soft-rank criterion requires a positive weight")
        if self.role is not CriterionRole.SOFT_RANK and self.weight is not None:
            raise ValueError("only soft-rank criterion may define weight")
        return self


class ScreeningProject(BaseModel):
    """Versioned screening contract; execution is allowed only after confirmation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    project_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    title: str = Field(min_length=1, max_length=200)
    research_question: str = Field(min_length=1, max_length=2000)
    application_context: str | None = Field(default=None, max_length=4000)
    material_scope: MaterialScope
    criteria: tuple[ScreeningCriterion, ...] = Field(min_length=1, max_length=50)
    resource_limits: ProjectResourceLimits = Field(
        default_factory=ProjectResourceLimits
    )
    status: ProjectStatus = ProjectStatus.DRAFT
    revision: int = Field(default=1, ge=1)
    confirmed_revision: int | None = Field(default=None, ge=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def confirmation_matches_status(self) -> Self:
        if (
            self.confirmed_revision is not None
            and self.confirmed_revision > self.revision
        ):
            raise ValueError("confirmed_revision cannot exceed revision")
        if self.status is ProjectStatus.DRAFT and self.confirmed_revision is not None:
            raise ValueError("draft project cannot retain a confirmed revision")
        return self


class DecisionLogEntry(BaseModel):
    """Immutable explanation of one persisted project decision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    project_id: str
    revision: int = Field(ge=1)
    action: str = Field(min_length=1, max_length=100)
    actor: str = Field(min_length=1, max_length=100)
    reason: str = Field(min_length=1, max_length=2000)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    details: dict[str, Any] = Field(default_factory=dict)
