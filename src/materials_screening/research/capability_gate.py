"""Deterministic capability assessment before any screening execution."""

from __future__ import annotations

import math
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from materials_screening.research.models import (
    CriterionRole,
    ScreeningCriterion,
    ScreeningProject,
)
from materials_screening.research.property_registry import (
    PropertyDefinition,
    PropertyRegistry,
    PropertyValueType,
)


class CapabilityStatus(StrEnum):
    SUPPORTED = "supported"
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"


class CapabilityIssue(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    criterion_index: int
    property_id: str
    message: str
    blocking: bool = True


class AssessedCriterion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    criterion_index: int
    requested_property: str
    canonical_property_id: str | None
    label_zh: str | None
    source_roles: tuple[str, ...] = ()
    comparability_notes: str | None = None


class ConfirmationCard(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    project_id: str
    revision: int
    question: str
    executable: bool
    supported_items: tuple[str, ...]
    blocking_items: tuple[str, ...]
    missing_value_statement: str
    confirmation_prompt: str


class CapabilityAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    project_id: str
    revision: int
    status: CapabilityStatus
    criteria: tuple[AssessedCriterion, ...]
    issues: tuple[CapabilityIssue, ...]
    confirmation_card: ConfirmationCard

    @property
    def executable(self) -> bool:
        return self.status is CapabilityStatus.SUPPORTED


class PropertyCapabilityGate:
    """Reject unknown, incompatible, or currently unqueryable properties."""

    def __init__(self, registry: PropertyRegistry) -> None:
        self._registry = registry

    def assess(self, project: ScreeningProject) -> CapabilityAssessment:
        issues: list[CapabilityIssue] = []
        assessed: list[AssessedCriterion] = []
        supported_items: list[str] = []
        for index, criterion in enumerate(project.criteria):
            definition = self._registry.resolve(criterion.property_id)
            assessed.append(self._assessed(index, criterion, definition))
            criterion_issues = self._issues(index, criterion, definition)
            issues.extend(criterion_issues)
            if not criterion_issues and definition is not None:
                supported_items.append(
                    f"{definition.label_zh}（{criterion.role.value}，"
                    f"缺失值：{criterion.missing_policy.value}）"
                )

        status = self._status(project, issues)
        blocking_items = tuple(issue.message for issue in issues if issue.blocking)
        card = ConfirmationCard(
            project_id=project.project_id,
            revision=project.revision,
            question=project.research_question,
            executable=status is CapabilityStatus.SUPPORTED,
            supported_items=tuple(supported_items),
            blocking_items=blocking_items,
            missing_value_statement=(
                "每个条件均使用已确认的缺失值策略；系统不会用其他属性或默认值静默替代。"
            ),
            confirmation_prompt=(
                "请确认以上筛选条件、来源角色和缺失值处理后再运行。"
                if status is CapabilityStatus.SUPPORTED
                else "请先修改阻塞条件；当前项目不能运行。"
            ),
        )
        return CapabilityAssessment(
            project_id=project.project_id,
            revision=project.revision,
            status=status,
            criteria=tuple(assessed),
            issues=tuple(issues),
            confirmation_card=card,
        )

    @staticmethod
    def _assessed(
        index: int,
        criterion: ScreeningCriterion,
        definition: PropertyDefinition | None,
    ) -> AssessedCriterion:
        return AssessedCriterion(
            criterion_index=index,
            requested_property=criterion.property_id,
            canonical_property_id=(
                None if definition is None else definition.property_id
            ),
            label_zh=None if definition is None else definition.label_zh,
            source_roles=(
                ()
                if definition is None
                else tuple(sorted(role.value for role in definition.source_roles))
            ),
            comparability_notes=(
                None if definition is None else definition.comparability_notes
            ),
        )

    @staticmethod
    def _status(
        project: ScreeningProject, issues: list[CapabilityIssue]
    ) -> CapabilityStatus:
        if not issues:
            return CapabilityStatus.SUPPORTED
        unknown_core = any(
            issue.code == "UNREGISTERED_PROPERTY"
            and project.criteria[issue.criterion_index].role
            in {CriterionRole.HARD_FILTER, CriterionRole.SOFT_RANK}
            for issue in issues
        )
        return (
            CapabilityStatus.UNSUPPORTED
            if unknown_core
            else CapabilityStatus.PARTIAL
        )

    @staticmethod
    def _issues(
        index: int,
        criterion: ScreeningCriterion,
        definition: PropertyDefinition | None,
    ) -> list[CapabilityIssue]:
        if definition is None:
            return [
                CapabilityIssue(
                    code="UNREGISTERED_PROPERTY",
                    criterion_index=index,
                    property_id=criterion.property_id,
                    message=(
                        f"属性“{criterion.property_id}”尚未登记，不能用于"
                        f"{criterion.role.value}，且不会自动替换为近似属性。"
                    ),
                )
            ]
        issues: list[CapabilityIssue] = []
        if criterion.role not in definition.supported_roles:
            issues.append(
                CapabilityIssue(
                    code="UNSUPPORTED_ROLE",
                    criterion_index=index,
                    property_id=definition.property_id,
                    message=f"“{definition.label_zh}”不支持 {criterion.role.value}。",
                )
            )
        if criterion.operator not in definition.supported_operators:
            issues.append(
                CapabilityIssue(
                    code="UNSUPPORTED_OPERATOR",
                    criterion_index=index,
                    property_id=definition.property_id,
                    message=(
                        f"“{definition.label_zh}”不支持操作符 "
                        f"{criterion.operator.value}。"
                    ),
                )
            )
        role_operators = {
            CriterionRole.HARD_FILTER: {
                "eq",
                "gte",
                "lte",
                "between",
            },
            CriterionRole.SOFT_RANK: {"target", "prefer_min", "prefer_max"},
            CriterionRole.REPORT_ONLY: {"eq"},
        }
        if criterion.operator.value not in role_operators[criterion.role]:
            issues.append(
                CapabilityIssue(
                    code="OPERATOR_ROLE_MISMATCH",
                    criterion_index=index,
                    property_id=definition.property_id,
                    message=(
                        f"操作符 {criterion.operator.value} 不能用于 "
                        f"{criterion.role.value}。"
                    ),
                )
            )
        if criterion.missing_policy not in definition.allowed_missing_policies:
            issues.append(
                CapabilityIssue(
                    code="UNSUPPORTED_MISSING_POLICY",
                    criterion_index=index,
                    property_id=definition.property_id,
                    message=f"“{definition.label_zh}”不允许该缺失值处理方式。",
                )
            )
        if criterion.unit != definition.canonical_unit:
            issues.append(
                CapabilityIssue(
                    code="UNIT_MISMATCH",
                    criterion_index=index,
                    property_id=definition.property_id,
                    message=(
                        f"“{definition.label_zh}”必须使用单位 "
                        f"{definition.canonical_unit or '无单位'}，当前为 "
                        f"{criterion.unit or '无单位'}。"
                    ),
                )
            )
        if (
            criterion.role is CriterionRole.HARD_FILTER
            and not definition.query_supported
        ):
            issues.append(
                CapabilityIssue(
                    code="QUERY_NOT_SUPPORTED",
                    criterion_index=index,
                    property_id=definition.property_id,
                    message=f"当前数据库适配器还不能按“{definition.label_zh}”执行硬过滤。",
                )
            )
        issues.extend(
            PropertyCapabilityGate._value_issues(index, criterion, definition)
        )
        return issues

    @staticmethod
    def _value_issues(
        index: int,
        criterion: ScreeningCriterion,
        definition: PropertyDefinition,
    ) -> list[CapabilityIssue]:
        values = criterion.values
        valid = True
        if definition.value_type is PropertyValueType.NUMBER:
            valid = all(
                isinstance(value, int | float)
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in values
            )
        elif definition.value_type is PropertyValueType.INTEGER:
            valid = all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in values
            )
        elif definition.value_type is PropertyValueType.BOOLEAN:
            valid = all(isinstance(value, bool) for value in values)
        elif definition.value_type in {
            PropertyValueType.TEXT,
            PropertyValueType.TEXT_SET,
        }:
            valid = all(
                isinstance(value, str) and bool(value.strip()) for value in values
            )
        if valid:
            return []
        return [
            CapabilityIssue(
                code="VALUE_TYPE_MISMATCH",
                criterion_index=index,
                property_id=definition.property_id,
                message=f"“{definition.label_zh}”的取值类型不符合注册定义。",
            )
        ]
