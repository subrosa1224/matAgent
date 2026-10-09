"""Inspection routing and evidence-backed explicit description reports."""

from __future__ import annotations

import json
import re

from pydantic import ValidationError

from materials_screening.agent.model_base import (
    AgentModelStatus,
    MaterialAgentModel,
    MaterialAgentRequest,
    MaterialAgentResponse,
)
from materials_screening.agent.models import (
    AgentFinalDraft,
    AgentFinalStatus,
    AgentFunctionCallItem,
    AgentFunctionOutputItem,
    AgentMessageItem,
    ToolResultEnvelope,
)

from .descriptive_report import render_descriptive_report
from .models import (
    AssessDataQualityInput,
    DataAnalysisToolPayload,
    DescribeDatasetInput,
    InspectDatasetInput,
)

_DATASET_ID = re.compile(r"dataset-[A-Za-z0-9][A-Za-z0-9._:-]{0,247}")


class DataAnalysisRoutingModel:
    """Keep explicit inspection routing and render real descriptive outputs."""

    def __init__(self, delegate: MaterialAgentModel) -> None:
        self._delegate = delegate

    def generate(self, request: MaterialAgentRequest) -> MaterialAgentResponse:
        message, outputs = _current_turn(request)
        multiple = _independent_description_inputs(message)
        if multiple is not None:
            return _independent_descriptions(request, message, outputs, multiple)
        report = _description_final(request, message, outputs)
        if report is not None:
            return MaterialAgentResponse(
                status=AgentModelStatus.COMPLETED,
                output_items=(
                    AgentMessageItem(
                        role="assistant", content=report.model_dump_json()
                    ),
                ),
                request_id="deterministic-description-report",
                provider="deterministic",
                model="data-analysis-report-v1",
            )
        explicit = _explicit_description_input(message)
        if explicit is not None:
            next_step = _description_next(request, explicit, outputs)
            if next_step is not None:
                return next_step
        match = _DATASET_ID.search(message)
        if match is None or outputs:
            return self._delegate.generate(request)
        if re.search(r"质量|缺失|重复|quality", message, re.IGNORECASE):
            name = "assess_data_quality"
        elif re.search(r"检查|预览|字段|inspect|preview", message, re.IGNORECASE):
            name = "inspect_dataset"
        else:
            return self._delegate.generate(request)
        return MaterialAgentResponse(
            status=AgentModelStatus.COMPLETED,
            output_items=(
                AgentFunctionCallItem(
                    call_id="deterministic-data-inspection",
                    name=name,
                    arguments=json.dumps({"dataset_id": match.group(0)}),
                ),
            ),
            request_id="deterministic-data-inspection",
            provider="deterministic",
            model="data-analysis-router-v1",
        )


def _explicit_description_input(message: str) -> DescribeDatasetInput | None:
    """Recognize complete, single-operation arguments without guessing fields."""

    dataset_ids = _DATASET_ID.findall(message)
    column_pattern = (
        r"\bcolumns\s*=\s*"
        r"([A-Za-z_][A-Za-z0-9_.]*(?:\s*,\s*[A-Za-z_][A-Za-z0-9_.]*)*)"
    )
    columns = re.search(column_pattern, message)
    grouping = re.search(r"\bgroup_by\s*=\s*([A-Za-z_][A-Za-z0-9_.]*)", message)
    intent = re.sub(
        column_pattern + r"|\bgroup_by\s*=\s*[A-Za-z0-9_.]+",
        "",
        _DATASET_ID.sub("", message),
    )
    if (
        re.search(r"\bgroup_by_each\s*=", message)
        or len(dataset_ids) != 1
        or columns is None
        or not re.search(r"描述|describe|descriptive", message, re.IGNORECASE)
        or re.search(
            r"相关|检验|清洗|转换|散点|箱线|直方|图表|绘图|画图|热力|导出|"
            r"生成报告|回归|correlat|test|transform|clean|plot|chart|export|"
            r"create.*report|scatter|histogram|heatmap|regression|anova",
            intent,
            re.IGNORECASE,
        )
    ):
        return None
    if (
        len(re.findall(r"\bcolumns\s*=", message)) != 1
        or len(re.findall(r"\bgroup_by\s*=", message)) > 1
        or message[columns.end() :].lstrip().startswith((",", "="))
        or (
            grouping is not None
            and message[grouping.end() :].lstrip().startswith((",", "="))
        )
        or re.search(r"\bquantiles\s*=", message)
        or (
            grouping is None
            and re.search(r"\bgroup_by\s*=|分组|按|grouped|group by", intent)
        )
    ):
        return None
    requested_columns = tuple(
        value.strip().rstrip(".") for value in columns.group(1).split(",")
    )
    if len(set(requested_columns)) != len(requested_columns):
        return None
    try:
        return DescribeDatasetInput(
            dataset_id=dataset_ids[0],
            columns=requested_columns,
            group_by=grouping.group(1).rstrip(".") if grouping else None,
        )
    except ValidationError:
        return None


def _current_calls(request: MaterialAgentRequest) -> dict[str, AgentFunctionCallItem]:
    """Never reuse a previous user turn's successful check or attempt."""

    items = request.input_items
    start = max(
        index
        for index, item in enumerate(items)
        if isinstance(item, AgentMessageItem) and item.role == "user"
    )
    return {
        item.call_id: item
        for item in items[start + 1 :]
        if isinstance(item, AgentFunctionCallItem)
    }


def _description_next(
    request: MaterialAgentRequest,
    arguments: DescribeDatasetInput,
    outputs: tuple[AgentFunctionOutputItem, ...],
    *,
    allow_other_descriptions: bool = False,
) -> MaterialAgentResponse | None:
    """Quality check -> real statistics, with no remote operation selection."""

    if not outputs:
        call = AgentFunctionCallItem(
            call_id="deterministic-data-inspection",
            name="assess_data_quality",
            arguments=json.dumps({"dataset_id": arguments.dataset_id}),
        )
    else:
        calls = _current_calls(request)
        if not allow_other_descriptions and any(
            item.name == "describe_dataset" for item in calls.values()
        ):
            return None  # No automatic retry of an attempted description.
        for output in reversed(outputs):
            inspection_call = calls.get(output.call_id)
            if inspection_call is None or inspection_call.name not in {
                "assess_data_quality",
                "inspect_dataset",
            }:
                continue
            input_model = (
                AssessDataQualityInput
                if inspection_call.name == "assess_data_quality"
                else InspectDatasetInput
            )
            try:
                checked = input_model.model_validate_json(inspection_call.arguments)
            except ValidationError:
                continue
            if checked.dataset_id != arguments.dataset_id:
                continue
            try:
                envelope = ToolResultEnvelope.model_validate_json(output.output)
                if (
                    envelope.call_id != inspection_call.call_id
                    or envelope.tool_name != inspection_call.name
                ):
                    raise ValueError("检查结果的调用引用不匹配")
                if envelope.status == "error":
                    return _description_routing_error(
                        f"数据质量检查失败：{envelope.error.message}；描述统计未执行。"
                    )
                payload = DataAnalysisToolPayload.model_validate(envelope.output)
                if (
                    payload.dataset is None
                    or payload.inspection is None
                    or payload.dataset.dataset_id != arguments.dataset_id
                    or payload.dataset != payload.inspection.dataset
                    or payload.evidence_id != envelope.evidence_id
                ):
                    raise ValueError("检查结果的数据集或证据引用不一致")
            except (ValueError, TypeError):
                return _description_routing_error(
                    "数据质量检查返回结果无效；描述统计未执行。"
                )
            call = AgentFunctionCallItem(
                call_id=(
                    f"deterministic-data-description-{arguments.group_by}"
                    if allow_other_descriptions
                    else "deterministic-data-description"
                ),
                name="describe_dataset",
                arguments=arguments.model_dump_json(),
            )
            break
        else:
            return None
    if not request.allow_tool_calls:
        return _description_routing_error(
            "本轮不允许继续调用统计工具；描述统计未执行。"
        )
    return MaterialAgentResponse(
        status=AgentModelStatus.COMPLETED,
        output_items=(call,),
        request_id=call.call_id,
        provider="deterministic",
        model="data-analysis-router-v2",
    )


def _description_routing_error(message: str) -> MaterialAgentResponse:
    draft = AgentFinalDraft(status=AgentFinalStatus.ERROR, answer=message)
    return MaterialAgentResponse(
        status=AgentModelStatus.COMPLETED,
        output_items=(
            AgentMessageItem(role="assistant", content=draft.model_dump_json()),
        ),
        request_id="deterministic-description-routing-error",
        provider="deterministic",
        model="data-analysis-router-v2",
    )


def _description_final(
    request: MaterialAgentRequest,
    message: str,
    outputs: tuple[AgentFunctionOutputItem, ...],
) -> AgentFinalDraft | None:
    """Only finalize an explicit, single descriptive task after its real tool."""

    requested = _explicit_description_input(message)
    if not outputs or requested is None:
        return None
    calls = _current_calls(request)
    for output in reversed(outputs):
        call = calls.get(output.call_id)
        if call is None or call.name != "describe_dataset":
            continue
        try:
            args = DescribeDatasetInput.model_validate_json(call.arguments)
        except ValidationError:
            continue
        if (
            args.dataset_id != requested.dataset_id
            or args.columns != requested.columns
            or args.group_by != requested.group_by
            or args.quantiles != requested.quantiles
        ):
            continue
        try:
            envelope = ToolResultEnvelope.model_validate_json(output.output)
            if envelope.call_id != call.call_id or envelope.tool_name != call.name:
                raise ValueError("工具返回的调用引用不匹配")
            if envelope.status == "error":
                return AgentFinalDraft(
                    status=AgentFinalStatus.ERROR,
                    answer=f"描述统计工具执行失败：{envelope.error.message}",
                )
            payload = DataAnalysisToolPayload.model_validate(envelope.output)
            analysis = payload.analysis
            if (
                analysis is None
                or payload.dataset is None
                or payload.evidence_id != envelope.evidence_id
                or analysis.evidence_id != envelope.evidence_id
                or analysis.dataset_id != args.dataset_id
                or analysis.parameters.get("columns") != list(args.columns)
                or analysis.parameters.get("group_by") != args.group_by
                or analysis.parameters.get("quantiles") != list(args.quantiles)
            ):
                raise ValueError("数据集、分析参数或证据引用不一致，或缺少总行数元数据")
            return render_descriptive_report(analysis, payload.dataset)
        except (ValueError, TypeError) as exc:
            # Do not send inconsistent statistics to a prose model to repair.
            detail = (
                str(exc) if not isinstance(exc, ValidationError) else "工具结果格式无效"
            )
            return AgentFinalDraft(
                status=AgentFinalStatus.ERROR,
                answer=f"统计结果一致性校验失败：{detail[:400]}。未生成统计结论。",
            )
    return None


def _independent_description_inputs(
    message: str,
) -> tuple[DescribeDatasetInput, ...] | None:
    """Explicit separate dimensions, not a composite-key group or model guess."""
    pattern = (
        r"\bgroup_by_each\s*=\s*"
        r"([A-Za-z_][A-Za-z0-9_.]*(?:\s*,\s*[A-Za-z_][A-Za-z0-9_.]*)*)"
    )
    match = re.search(pattern, message)
    if (
        match is None
        or len(re.findall(r"\bgroup_by_each\s*=", message)) != 1
        or re.search(r"\bgroup_by\s*=", message)
        or message[match.end() :].lstrip().startswith((",", "="))
    ):
        return None
    names = tuple(name.strip().rstrip(".") for name in match.group(1).split(","))
    if not 2 <= len(names) <= 3 or len(set(names)) != len(names):
        return None
    arguments = tuple(
        _explicit_description_input(
            message[: match.start()] + f"group_by={name}" + message[match.end() :]
        )
        for name in names
    )
    if any(item is None for item in arguments):
        return None
    return tuple(item for item in arguments if item is not None)


def _independent_descriptions(
    request: MaterialAgentRequest,
    message: str,
    outputs: tuple[AgentFunctionOutputItem, ...],
    arguments: tuple[DescribeDatasetInput, ...],
) -> MaterialAgentResponse:
    reports: list[AgentFinalDraft] = []
    calls = _current_calls(request)
    for args in arguments:
        single_message = re.sub(
            r"\bgroup_by_each\s*=\s*[A-Za-z_][A-Za-z0-9_.]*(?:\s*,\s*[A-Za-z_][A-Za-z0-9_.]*)*",
            f"group_by={args.group_by}",
            message,
        )
        report = _description_final(request, single_message, outputs)
        if report is not None:
            if report.status != AgentFinalStatus.COMPLETED:
                return _description_routing_error(report.answer)
            reports.append(report)
            continue
        for call in calls.values():
            if call.name != "describe_dataset":
                continue
            try:
                attempted = DescribeDatasetInput.model_validate_json(call.arguments)
            except ValidationError:
                continue
            if attempted == args:
                return _description_routing_error(
                    f"{args.group_by} 的统计已尝试但缺少有效返回；"
                    "不自动重复，也不宣称全部分组完成。"
                )
        next_step = _description_next(
            request, args, outputs, allow_other_descriptions=True
        )
        return next_step or _description_routing_error(
            "缺少本轮有效质量检查；未完成全部分组。"
        )
    warnings = list(
        dict.fromkeys(note for report in reports for note in report.warnings)
    )
    sections = []
    budget = 7200 // len(reports)
    for args, report in zip(arguments, reports, strict=True):
        answer = report.answer
        if len(answer) > budget:
            answer = answer[:budget].rsplit("\n", 1)[0]
            note = f"{args.group_by} 仅展示部分统计；完整结果见该分组的分析 ID。"
            warnings.append(note)
            answer += "\n" + note
        sections.append(f"独立分组：{args.group_by}\n{answer}")
    draft = AgentFinalDraft(
        status=AgentFinalStatus.COMPLETED,
        answer="\n\n".join(sections),
        evidence_ids=list(
            dict.fromkeys(e for report in reports for e in report.evidence_ids)
        ),
        warnings=warnings,
    )
    return MaterialAgentResponse(
        status=AgentModelStatus.COMPLETED,
        output_items=(
            AgentMessageItem(role="assistant", content=draft.model_dump_json()),
        ),
        request_id="deterministic-independent-descriptions",
        provider="deterministic",
        model="data-analysis-report-v2",
    )


def _current_turn(
    request: MaterialAgentRequest,
) -> tuple[str, tuple[AgentFunctionOutputItem, ...]]:
    items = tuple(request.input_items)
    for index in range(len(items) - 1, -1, -1):
        item = items[index]
        if isinstance(item, AgentMessageItem) and item.role == "user":
            outputs = tuple(
                value
                for value in items[index + 1 :]
                if isinstance(value, AgentFunctionOutputItem)
            )
            return item.content, outputs
    return "", ()
