"""Conservative material-identity rules for cross-source evidence use."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pymatgen.core.composition import Composition

from materials_screening.chemistry import normalize_chemsys


class IdentityUse(StrEnum):
    CANDIDATE_PROPERTY = "candidate_property"
    BACKGROUND_ONLY = "background_only"
    BLOCKED = "blocked"


class IdentityRelation(StrEnum):
    EXACT_SOURCE_ID = "exact_source_id"
    STRUCTURE_MATCH = "structure_match"
    SAME_FORMULA_DIFFERENT_PHASE = "same_formula_different_phase"
    SAME_FORMULA_INCOMPLETE_STRUCTURE = "same_formula_incomplete_structure"
    RELATED_MATERIAL_FAMILY = "related_material_family"
    SAME_CHEMSYS_ONLY = "same_chemsys_only"
    NAME_ONLY = "name_only"
    NO_MATCH = "no_match"


class MaterialIdentityRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_kind: str = Field(min_length=1, max_length=100)
    source_material_id: str | None = Field(default=None, max_length=300)
    formula: str | None = Field(default=None, max_length=300)
    chemsys: str | None = Field(default=None, max_length=300)
    material_name: str | None = Field(default=None, max_length=500)
    structure_fingerprint: str | None = Field(default=None, max_length=100)
    space_group_number: int | None = Field(default=None, ge=1, le=230)
    material_family: str | None = Field(default=None, max_length=300)
    composition_variant: str = Field(default="ideal", max_length=100)
    phase_conditions: tuple[str, ...] = Field(default=(), max_length=30)


class IdentityAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    relation: IdentityRelation
    identity_use: IdentityUse
    same_structure_candidate: bool
    requires_human_review: bool
    reasons: tuple[str, ...]


class MaterialIdentityAssessmentRecord(BaseModel):
    """Persistable link between one candidate and one external evidence record."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    identity_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,247}$")
    project_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,247}$")
    candidate_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,247}$")
    evidence_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,247}$")
    left: MaterialIdentityRecord
    right: MaterialIdentityRecord
    relation: IdentityRelation
    identity_use: IdentityUse
    same_structure_candidate: bool
    requires_human_review: bool
    reasons: tuple[str, ...]
    review_status: Literal["validated", "pending_domain_review", "human_reviewed"]

    @model_validator(mode="after")
    def identity_use_is_consistent(self) -> MaterialIdentityAssessmentRecord:
        if self.same_structure_candidate != (
            self.identity_use is IdentityUse.CANDIDATE_PROPERTY
        ):
            raise ValueError(
                "candidate-property identity use requires a same-structure candidate"
            )
        if self.requires_human_review and self.review_status == "validated":
            raise ValueError("human-review requirement cannot be marked rule-validated")
        return self


class MaterialIdentityResolver:
    """Resolve identity without equating formula or name with structure."""

    def resolve(
        self, left: MaterialIdentityRecord, right: MaterialIdentityRecord
    ) -> IdentityAssessment:
        if (
            left.source_kind == right.source_kind
            and left.source_material_id
            and left.source_material_id == right.source_material_id
        ):
            return self._exact(
                IdentityRelation.EXACT_SOURCE_ID,
                "同一来源中的有效材料 ID 相同。",
                left,
                right,
            )

        left_formula = _formula(left.formula)
        right_formula = _formula(right.formula)
        same_formula = bool(left_formula and left_formula == right_formula)
        if (
            same_formula
            and left.structure_fingerprint
            and left.structure_fingerprint == right.structure_fingerprint
            and left.space_group_number is not None
            and left.space_group_number == right.space_group_number
        ):
            return self._exact(
                IdentityRelation.STRUCTURE_MATCH,
                "规范化化学式、空间群和结构指纹均匹配。",
                left,
                right,
            )

        if same_formula:
            if (
                left.space_group_number is not None
                and right.space_group_number is not None
                and left.space_group_number != right.space_group_number
            ):
                return IdentityAssessment(
                    relation=IdentityRelation.SAME_FORMULA_DIFFERENT_PHASE,
                    identity_use=IdentityUse.BLOCKED,
                    same_structure_candidate=False,
                    requires_human_review=True,
                    reasons=("化学式相同但空间群不同，必须保留为不同物相。",),
                )
            return IdentityAssessment(
                relation=IdentityRelation.SAME_FORMULA_INCOMPLETE_STRUCTURE,
                identity_use=IdentityUse.BACKGROUND_ONLY,
                same_structure_candidate=False,
                requires_human_review=True,
                reasons=("化学式相同，但结构信息不足以确认同一物相。",),
            )

        if (
            left.material_family
            and right.material_family
            and _text(left.material_family) == _text(right.material_family)
        ):
            return IdentityAssessment(
                relation=IdentityRelation.RELATED_MATERIAL_FAMILY,
                identity_use=IdentityUse.BACKGROUND_ONLY,
                same_structure_candidate=False,
                requires_human_review=True,
                reasons=("仅能确认属于相同材料家族或母相关系。",),
            )

        left_chemsys = _chemsys(left.chemsys)
        right_chemsys = _chemsys(right.chemsys)
        if left_chemsys and left_chemsys == right_chemsys:
            return IdentityAssessment(
                relation=IdentityRelation.SAME_CHEMSYS_ONLY,
                identity_use=IdentityUse.BACKGROUND_ONLY,
                same_structure_candidate=False,
                requires_human_review=False,
                reasons=("只共享元素体系，不能映射为同一材料。",),
            )

        if (
            left.material_name
            and right.material_name
            and _text(left.material_name) == _text(right.material_name)
        ):
            return IdentityAssessment(
                relation=IdentityRelation.NAME_ONLY,
                identity_use=IdentityUse.BACKGROUND_ONLY,
                same_structure_candidate=False,
                requires_human_review=True,
                reasons=("仅材料名称相同，不能据此合并身份或数值。",),
            )
        return IdentityAssessment(
            relation=IdentityRelation.NO_MATCH,
            identity_use=IdentityUse.BLOCKED,
            same_structure_candidate=False,
            requires_human_review=False,
            reasons=("没有足以建立材料映射的组成或结构证据。",),
        )

    @staticmethod
    def _exact(
        relation: IdentityRelation,
        reason: str,
        left: MaterialIdentityRecord,
        right: MaterialIdentityRecord,
    ) -> IdentityAssessment:
        condition_note: tuple[str, ...] = ()
        needs_review = False
        if set(left.phase_conditions) != set(right.phase_conditions):
            condition_note = (
                "相或样品条件不同；身份可关联，但属性仍需单独检查可比性。",
            )
            needs_review = True
        return IdentityAssessment(
            relation=relation,
            identity_use=IdentityUse.CANDIDATE_PROPERTY,
            same_structure_candidate=True,
            requires_human_review=needs_review,
            reasons=(reason, *condition_note),
        )


def _formula(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return Composition(value).reduced_formula
    except (TypeError, ValueError):
        return None


def _chemsys(value: str | None) -> str | None:
    try:
        return normalize_chemsys(value)
    except ValueError:
        return None


def _text(value: str) -> str:
    return " ".join(value.casefold().split())
