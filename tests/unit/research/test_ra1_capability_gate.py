from __future__ import annotations

import math

import pytest

from materials_screening.research import (
    CapabilityStatus,
    CriterionOperator,
    CriterionRole,
    MaterialScope,
    MissingValuePolicy,
    ProjectResourceLimits,
    PropertyCapabilityGate,
    PropertyDefinition,
    PropertyRegistry,
    PropertyValueType,
    ResearchProjectError,
    ScreeningCriterion,
    ScreeningProject,
    SourceRole,
    default_property_registry,
)


def _criterion(
    property_id: str = "band_gap_ev",
    *,
    role: CriterionRole = CriterionRole.HARD_FILTER,
    operator: CriterionOperator = CriterionOperator.GTE,
    values: tuple[str | float | int | bool, ...] = (3.0,),
    unit: str | None = "eV",
    missing_policy: MissingValuePolicy = MissingValuePolicy.EXCLUDE,
) -> ScreeningCriterion:
    return ScreeningCriterion(
        property_id=property_id,
        role=role,
        operator=operator,
        values=values,
        unit=unit,
        missing_policy=missing_policy,
        weight=1.0 if role is CriterionRole.SOFT_RANK else None,
    )


def _project(*criteria: ScreeningCriterion) -> ScreeningProject:
    return ScreeningProject(
        project_id="project-ra1-test",
        title="宽禁带氧化物筛选",
        research_question="哪些晶态无机氧化物满足确认的筛选条件？",
        material_scope=MaterialScope(required_elements=("O",)),
        criteria=criteria or (_criterion(),),
    )


def test_registered_alias_resolves_only_to_declared_property() -> None:
    registry = default_property_registry()

    assert registry.require("禁带宽度").property_id == "band_gap_ev"
    assert registry.resolve("hardness") is None


def test_registry_rejects_alias_collision() -> None:
    base = dict(
        value_type=PropertyValueType.NUMBER,
        supported_roles=frozenset({CriterionRole.REPORT_ONLY}),
        supported_operators=frozenset({CriterionOperator.EQ}),
        source_roles=frozenset({SourceRole.USER_DATA}),
        allowed_missing_policies=frozenset(
            {MissingValuePolicy.KEEP_WITH_WARNING}
        ),
        query_supported=False,
        comparability_notes="test-only definition",
    )
    first = PropertyDefinition(
        property_id="first_property",
        label_zh="属性甲",
        aliases=("shared",),
        **base,
    )
    second = PropertyDefinition(
        property_id="second_property",
        label_zh="属性乙",
        aliases=("shared",),
        **base,
    )

    with pytest.raises(ResearchProjectError, match="belongs to both") as exc_info:
        PropertyRegistry((first, second))

    assert exc_info.value.code == "ALIAS_COLLISION"


@pytest.mark.parametrize("role", [CriterionRole.HARD_FILTER, CriterionRole.SOFT_RANK])
def test_unregistered_property_cannot_filter_or_rank(role: CriterionRole) -> None:
    assessment = PropertyCapabilityGate(default_property_registry()).assess(
        _project(_criterion("breakdown_strength", role=role, unit="MV/cm"))
    )

    assert assessment.status is CapabilityStatus.UNSUPPORTED
    assert assessment.criteria[0].canonical_property_id is None
    assert assessment.issues[0].code == "UNREGISTERED_PROPERTY"
    assert "不会自动替换" in assessment.issues[0].message
    assert not assessment.confirmation_card.executable


def test_currently_unqueryable_field_is_explicitly_blocked() -> None:
    assessment = PropertyCapabilityGate(default_property_registry()).assess(
        _project(
            _criterion(
                "is_gap_direct",
                operator=CriterionOperator.EQ,
                values=(True,),
                unit=None,
            )
        )
    )

    assert assessment.status is CapabilityStatus.PARTIAL
    assert {issue.code for issue in assessment.issues} == {"QUERY_NOT_SUPPORTED"}


def test_supported_property_exposes_source_and_scientific_boundary() -> None:
    assessment = PropertyCapabilityGate(default_property_registry()).assess(_project())

    assert assessment.status is CapabilityStatus.SUPPORTED
    assert assessment.executable
    assert assessment.criteria[0].canonical_property_id == "band_gap_ev"
    assert assessment.criteria[0].source_roles == ("database_calculated",)
    assert "不能直接替代实验光学带隙" in (
        assessment.criteria[0].comparability_notes or ""
    )
    assert "不会用其他属性" in assessment.confirmation_card.missing_value_statement


def test_unit_and_value_type_are_not_silently_coerced() -> None:
    assessment = PropertyCapabilityGate(default_property_registry()).assess(
        _project(_criterion(values=("3.0",), unit="meV"))
    )

    assert {issue.code for issue in assessment.issues} == {
        "UNIT_MISMATCH",
        "VALUE_TYPE_MISMATCH",
    }


def test_filter_cannot_use_ranking_operator() -> None:
    assessment = PropertyCapabilityGate(default_property_registry()).assess(
        _project(_criterion(operator=CriterionOperator.TARGET))
    )

    assert "OPERATOR_ROLE_MISMATCH" in {
        issue.code for issue in assessment.issues
    }


def test_non_finite_numeric_value_is_blocked() -> None:
    assessment = PropertyCapabilityGate(default_property_registry()).assess(
        _project(_criterion(values=(math.nan,)))
    )

    assert "VALUE_TYPE_MISMATCH" in {issue.code for issue in assessment.issues}


def test_missing_policy_is_required_by_contract() -> None:
    with pytest.raises(ValueError, match="missing_policy"):
        ScreeningCriterion.model_validate(
            {
                "property_id": "band_gap_ev",
                "role": "hard_filter",
                "operator": "gte",
                "values": [3.0],
                "unit": "eV",
            }
        )


def test_between_requires_exactly_two_values() -> None:
    with pytest.raises(ValueError, match="requires 2 value"):
        _criterion(operator=CriterionOperator.BETWEEN, values=(1.0,))


def test_material_scope_rejects_required_excluded_overlap() -> None:
    with pytest.raises(ValueError, match="overlap"):
        MaterialScope(required_elements=("O", "Fe"), excluded_elements=("fe",))


def test_resource_limits_keep_deep_review_inside_quick_review() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        ProjectResourceLimits(
            quick_literature_limit=2,
            deep_literature_limit=3,
        )
