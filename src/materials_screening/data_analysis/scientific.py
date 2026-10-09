"""Research-task contracts and bounded variable-role proposals."""

from __future__ import annotations

import re
import uuid
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from materials_screening.data_analysis.dataset_store import DatasetStore
from materials_screening.data_analysis.service import DataAnalysisService

ScientificDomain = Literal["general", "materials_science"]
ExperimentalDesign = Literal[
    "exploratory",
    "independent_groups",
    "paired",
    "repeated_measures",
    "continuous_relationship",
]

_BRIEF_ID = r"^brief-[a-f0-9]{32}$"
_IDENTIFIER_HINTS = ("id", "sample", "specimen", "subject", "batch", "编号", "样品")
_GROUP_HINTS = ("group", "method", "treatment", "condition", "组", "方法", "处理")
_MATERIAL_RESPONSE_HINTS = (
    "hardness",
    "strength",
    "conductivity",
    "density",
    "porosity",
    "band_gap",
    "modulus",
    "硬度",
    "强度",
    "电导",
    "密度",
    "孔隙",
)


class ScientificAnalysisProposal(BaseModel):
    """Unconfirmed role candidates inferred from a bounded dataset profile."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: str
    domain: ScientificDomain
    response_candidates: tuple[str, ...] = Field(default=(), max_length=50)
    group_candidates: tuple[str, ...] = Field(default=(), max_length=20)
    identifier_candidates: tuple[str, ...] = Field(default=(), max_length=20)
    covariate_candidates: tuple[str, ...] = Field(default=(), max_length=50)
    rationale: tuple[str, ...] = Field(default=(), max_length=20)
    requires_user_confirmation: bool = True

    @model_validator(mode="after")
    def require_confirmation(self) -> Self:
        if not self.requires_user_confirmation:
            raise ValueError("role proposals must require user confirmation")
        return self


class ScientificAnalysisBrief(BaseModel):
    """User-confirmed scientific question, design, and variable roles."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    brief_id: str = Field(default_factory=lambda: f"brief-{uuid.uuid4().hex}")
    dataset_id: str
    domain: ScientificDomain = "general"
    research_question: str = Field(min_length=5, max_length=2000)
    observation_unit: str = Field(min_length=1, max_length=256)
    design: ExperimentalDesign
    response_variables: tuple[str, ...] = Field(min_length=1, max_length=20)
    group_variable: str | None = Field(default=None, min_length=1, max_length=256)
    covariates: tuple[str, ...] = Field(default=(), max_length=20)
    subject_id_variable: str | None = Field(
        default=None, min_length=1, max_length=256
    )
    hypothesis: str | None = Field(default=None, min_length=3, max_length=2000)
    alpha: float = Field(default=0.05, gt=0, lt=1)
    practical_thresholds: dict[str, float] = Field(default_factory=dict)
    units: dict[str, str] = Field(default_factory=dict)
    data_source: str = Field(default="user_upload", min_length=1, max_length=256)
    roles_confirmed: bool = False

    @field_validator("brief_id")
    @classmethod
    def validate_brief_id(cls, value: str) -> str:
        if not re.fullmatch(_BRIEF_ID, value):
            raise ValueError("brief_id is invalid")
        return value

    @field_validator("response_variables", "covariates")
    @classmethod
    def validate_unique_columns(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) != len(set(values)) or any(not item.strip() for item in values):
            raise ValueError("variable lists must contain unique non-blank names")
        return values

    @field_validator("practical_thresholds")
    @classmethod
    def validate_thresholds(cls, values: dict[str, float]) -> dict[str, float]:
        if any(not key.strip() or value < 0 for key, value in values.items()):
            raise ValueError(
                "practical thresholds require names and non-negative values"
            )
        return values

    @model_validator(mode="after")
    def validate_design_and_roles(self) -> Self:
        assigned = set(self.response_variables)
        if self.group_variable in assigned:
            raise ValueError("group variable cannot also be a response variable")
        if assigned.intersection(self.covariates):
            raise ValueError("responses and covariates must be distinct")
        if self.subject_id_variable in assigned or self.subject_id_variable in set(
            self.covariates
        ):
            raise ValueError("subject ID cannot be a response or covariate")
        if self.roles_confirmed:
            if self.design == "independent_groups" and self.group_variable is None:
                raise ValueError("independent-group design requires a group variable")
            if self.design in {"paired", "repeated_measures"} and (
                self.subject_id_variable is None
            ):
                raise ValueError("paired/repeated design requires a subject ID")
            if self.design in {"paired", "repeated_measures"} and (
                self.group_variable is None
            ):
                raise ValueError(
                    "paired/repeated design requires a condition variable"
                )
            if self.design == "continuous_relationship" and not self.covariates:
                raise ValueError(
                    "continuous relationship design requires a predictor"
                )
        unknown_thresholds = set(self.practical_thresholds).difference(
            self.response_variables
        )
        if unknown_thresholds:
            raise ValueError("practical thresholds must reference response variables")
        return self


class ScientificBriefService:
    """Propose roles without converting a proposal into confirmed research intent."""

    def __init__(self, store: DatasetStore) -> None:
        self._inspection = DataAnalysisService(store)

    def propose_roles(
        self,
        dataset_id: str,
        *,
        domain: ScientificDomain = "materials_science",
    ) -> ScientificAnalysisProposal:
        inspection = self._inspection.inspect_dataset(dataset_id, limit=1)
        row_count = max(inspection.dataset.row_count, 1)
        numeric = [
            column.name
            for column in inspection.columns
            if column.inferred_type == "numeric" and column.unique_count > 1
        ]
        identifiers = [
            column.name
            for column in inspection.columns
            if _matches_hint(column.name, _IDENTIFIER_HINTS)
            or column.unique_count / row_count >= 0.9
        ]
        groups = [
            column.name
            for column in inspection.columns
            if 1 < column.unique_count <= 20
            and column.unique_count / row_count <= 0.5
            and (
                column.inferred_type in {"categorical", "string"}
                or _matches_hint(column.name, _GROUP_HINTS)
            )
        ]
        preferred = (
            [
                name
                for name in numeric
                if _matches_hint(name, _MATERIAL_RESPONSE_HINTS)
            ]
            if domain == "materials_science"
            else []
        )
        responses = [*preferred, *(name for name in numeric if name not in preferred)]
        covariates = [name for name in numeric if name not in identifiers]
        rationale = [
            "候选角色仅依据字段名、数据类型和基数提出，尚未代表研究意图。",
            "响应变量、分组方式和实验设计必须由用户确认后才能运行推断检验。",
        ]
        if domain == "materials_science" and preferred:
            rationale.append("材料属性字段被优先列为响应候选，但不据此推断材料机理。")
        return ScientificAnalysisProposal(
            dataset_id=dataset_id,
            domain=domain,
            response_candidates=tuple(responses[:50]),
            group_candidates=tuple(dict.fromkeys(groups[:20])),
            identifier_candidates=tuple(dict.fromkeys(identifiers[:20])),
            covariate_candidates=tuple(dict.fromkeys(covariates[:50])),
            rationale=tuple(rationale),
        )

    @staticmethod
    def require_confirmed(brief: ScientificAnalysisBrief) -> None:
        if not brief.roles_confirmed:
            raise ValueError(
                "research variable roles must be confirmed before inferential analysis"
            )


def _matches_hint(name: str, hints: tuple[str, ...]) -> bool:
    lowered = name.casefold()
    return any(hint.casefold() in lowered for hint in hints)
