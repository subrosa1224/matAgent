"""Conservative coverage of explicitly specified metrics, not scientific approval.

Call only with rows surviving the caller's evidence and sample-binding gates.
Metric names (including test conditions) and aliases must be specified explicitly;
this checker neither infers synonyms nor supplies missing measurements.
"""

import html
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class RequiredMetric(BaseModel):
    """An exact metric/explicit alias, optionally restricted by unit and group.

    No group restriction means at least one row, not complete coverage of all samples.
    Conditions such as cycle count must be part of the metric name or explicit alias.
    Units are case-sensitive and are not converted. Reference values are not accepted.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    metric: str = Field(min_length=1, max_length=120)
    aliases: tuple[str, ...] = Field(default=(), max_length=20)
    unit: str | None = Field(default=None, min_length=1, max_length=80)
    group_id: str | None = Field(default=None, min_length=1, max_length=120)

    @field_validator("metric", "unit", "group_id")
    @classmethod
    def reject_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("Requirement fields must not be blank")
        return value

    @field_validator("aliases")
    @classmethod
    def validate_aliases(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not value.strip() or len(value) > 120 for value in values):
            raise ValueError("Each explicit alias must contain 1 to 120 characters")
        return values


class MeasurementMetricFields(Protocol):
    """Read-only projection; caller remains responsible for evidence validation."""

    @property
    def measurement_id(self) -> str: ...

    @property
    def metric(self) -> str: ...

    @property
    def unit(self) -> str | None: ...

    @property
    def group_id(self) -> str: ...


class TaskMetricRequirements(BaseModel):
    """Explicit task policy, shared by extraction and the evidence-trial entry point."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    topic: str = Field(min_length=1, max_length=4000)
    required_metrics: tuple[RequiredMetric, ...] = Field(min_length=1, max_length=50)

    @field_validator("topic")
    @classmethod
    def normalize_topic(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("Task topic must not be blank")
        return normalized

    def ensure_topic(self, topic: str) -> None:
        if " ".join(topic.split()) != self.topic:
            raise ValueError("必需指标清单与当前研究问题不匹配")


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate policy key")
        result[key] = value
    return result


def load_task_metric_requirements(path: Path) -> TaskMetricRequirements:
    """Read a bounded JSON policy; never expose malformed input in an error."""
    try:
        with path.open("rb") as stream:
            raw = stream.read(65537)
        if len(raw) > 65536:
            raise ValueError("Policy exceeds size limit")
        payload = json.loads(
            raw.decode("utf-8-sig"), object_pairs_hook=_unique_json_object
        )
        return TaskMetricRequirements.model_validate(payload)
    except (OSError, UnicodeError, ValueError, ValidationError) as exc:
        raise ValueError(
            "无法读取必需指标清单：需要不超过64 KiB的UTF-8 JSON，"
            "含非空topic和1至50项required_metrics，不接受参考答案字段"
        ) from exc


class RequiredMetricCheck(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    requirement: RequiredMetric
    status: Literal["present", "missing"]
    measurement_ids: tuple[str, ...] = ()


class RequiredMetricCoverage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["not_checked", "covered", "incomplete"] = "not_checked"
    scope: Literal["specified_metrics_in_validated_measurements"] = (
        "specified_metrics_in_validated_measurements"
    )
    checks: tuple[RequiredMetricCheck, ...] = ()


def _metric_key(value: str) -> str:
    return " ".join(value.split()).casefold()


def check_required_metric_coverage(
    measurements: Sequence[MeasurementMetricFields],
    requirements: Sequence[RequiredMetric] = (),
) -> RequiredMetricCoverage:
    """Observe final rows without modifying evidence, values, units, or review status.

    Missing means no matching validated row, NOT that the paper reports no such data.
    Covered means only the specified checks passed, NOT that extraction is exhaustive.
    """
    if not requirements:
        return RequiredMetricCoverage()
    checks = []
    for requirement in requirements:
        names = {
            _metric_key(name) for name in (requirement.metric, *requirement.aliases)
        }
        matches = tuple(
            row.measurement_id
            for row in measurements
            if _metric_key(row.metric) in names
            and (requirement.unit is None or row.unit == requirement.unit)
            and (requirement.group_id is None or row.group_id == requirement.group_id)
        )
        checks.append(
            RequiredMetricCheck(
                requirement=requirement,
                status="present" if matches else "missing",
                measurement_ids=matches,
            )
        )
    return RequiredMetricCoverage(
        status="covered"
        if all(check.status == "present" for check in checks)
        else "incomplete",
        checks=tuple(checks),
    )


def _markdown_label(value: str) -> str:
    escaped = html.escape(" ".join(value.split()))
    return re.sub(r"([\\`*_{}\[\]!|])", r"\\\1", escaped)


def render_required_metric_coverage(
    coverage: RequiredMetricCoverage,
    *,
    scope_label: str = "最终保留的测量记录",
) -> str:
    """Deterministic visible status, independent of optional model narratives."""
    lines = ["## 必需指标覆盖检查", "", f"检查范围：{scope_label}。", ""]
    if coverage.status == "not_checked":
        lines.append("状态：未检查。未提供必需指标清单，不能据此宣称关键指标齐全。")
        return "\n".join(lines)
    if coverage.status == "incomplete":
        lines.append("状态：不完整。当前关键结果不完整，不能宣称科研问题已获完整回答。")
    else:
        lines.append("状态：已覆盖，仅表示指定要求有匹配记录。")
    lines.append("")
    for check in coverage.checks:
        requirement = check.requirement
        label = _markdown_label(requirement.metric)
        if requirement.unit is not None:
            label += f"；单位：{_markdown_label(requirement.unit)}"
        if requirement.group_id is not None:
            label += f"；样本编号：{_markdown_label(requirement.group_id)}"
        state = "已覆盖" if check.status == "present" else "缺失"
        lines.append(f"- {state}：{label}（匹配{len(check.measurement_ids)}条）。")
    lines.extend(
        (
            "",
            "缺失仅表示当前纳入记录未覆盖，不证明论文没有报告该数据；"
            "本检查不自动补提或补造数据，也不代表专家审核或科研结论验证通过。",
        )
    )
    if any(check.requirement.group_id is None for check in coverage.checks):
        lines.append("未限定样本的要求只检查至少一条匹配，不代表所有样本均已覆盖。")
    return "\n".join(lines)
