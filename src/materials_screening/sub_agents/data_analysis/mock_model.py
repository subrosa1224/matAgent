"""Offline model that exercises data-analysis tools without network access."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping

from materials_screening.agent.model_base import (
    AgentModelStatus,
    MaterialAgentRequest,
    MaterialAgentResponse,
)
from materials_screening.agent.models import (
    AgentFunctionCallItem,
    AgentFunctionOutputItem,
    AgentMessageItem,
)

_DATASET_ID = re.compile(r"dataset-[A-Za-z0-9][A-Za-z0-9._:-]{0,247}")
_ANALYSIS_ID = re.compile(r"analysis-[A-Za-z0-9][A-Za-z0-9._:-]{0,246}")
_ARTIFACT_ID = re.compile(r"artifact-[A-Za-z0-9][A-Za-z0-9._:-]{0,246}")


class DataAnalysisMockModel:
    def generate(self, request: MaterialAgentRequest) -> MaterialAgentResponse:
        outputs = [
            item
            for item in request.input_items
            if isinstance(item, AgentFunctionOutputItem)
        ]
        if outputs:
            envelope = json.loads(outputs[-1].output)
            payload = envelope.get("output", envelope)
            evidence = envelope.get("evidence_id") or payload.get("evidence_id")
            answer = _summarize_payload(payload)
            content = {
                "status": "completed",
                "answer": answer,
                "referenced_material_ids": [],
                "evidence_ids": [evidence] if evidence else [],
                "warnings": [],
                "follow_up_question": None,
            }
            return MaterialAgentResponse(
                status=AgentModelStatus.COMPLETED,
                output_items=(
                    AgentMessageItem(
                        role="assistant",
                        content=json.dumps(content, ensure_ascii=False),
                    ),
                ),
                request_id="mock-data-analysis-final",
                provider="mock",
                model="mock-data-analysis",
            )
        message = _last_user_message(request)
        match = _DATASET_ID.search(message)
        if match is None:
            content = {
                "status": "needs_user_input",
                "answer": "请提供已登记数据集的 dataset_id。",
                "referenced_material_ids": [],
                "evidence_ids": [],
                "warnings": [],
                "follow_up_question": "要分析哪个 dataset_id？",
            }
            return _message_response(content)
        routed = _route_tool(message, match.group(0), _all_text(request))
        if isinstance(routed, str):
            return _message_response(
                {
                    "status": "needs_user_input",
                    "answer": routed,
                    "referenced_material_ids": [],
                    "evidence_ids": [],
                    "warnings": [],
                    "follow_up_question": routed,
                }
            )
        name, arguments = routed
        return MaterialAgentResponse(
            status=AgentModelStatus.COMPLETED,
            output_items=(
                AgentFunctionCallItem(
                    call_id="mock-data-analysis-tool",
                    name=name,
                    arguments=json.dumps(arguments, ensure_ascii=False),
                ),
            ),
            request_id="mock-data-analysis-call",
            provider="mock",
            model="mock-data-analysis",
        )


def _summarize_payload(payload: dict[str, object]) -> str:
    inspection = payload.get("inspection")
    if isinstance(inspection, dict):
        dataset = inspection.get("dataset") or {}
        if isinstance(dataset, dict):
            return (
                f"数据集检查完成：{dataset.get('row_count', '—')} 行、"
                f"{dataset.get('column_count', '—')} 列。"
            )
    analysis = payload.get("analysis")
    if isinstance(analysis, dict):
        heading = (
            f"分析已完成：{analysis.get('analysis_type', 'analysis')}，"
            f"分析 ID 为 `{analysis.get('analysis_id', '—')}`。"
        )
        summary = analysis.get("summary")
        statistics = summary.get("statistics") if isinstance(summary, dict) else None
        if not isinstance(statistics, list) or not statistics:
            return heading
        lines = [heading, "按“指标 + 单位”分组的描述统计："]
        for item in statistics[:12]:
            if not isinstance(item, dict):
                continue
            column = item.get("column", "未命名指标")
            group = item.get("group")
            label = (
                str(group)
                if column == "numeric_value" and group is not None
                else f"{column} [{group}]"
                if group is not None
                else str(column)
            )
            lines.append(
                "- "
                f"{label}："
                f"n={item.get('count', '—')}，"
                f"均值={_display_number(item.get('mean'))}，"
                f"范围={_display_number(item.get('min'))}–"
                f"{_display_number(item.get('max'))}"
            )
        return "\n".join(lines)
    transform = payload.get("transform")
    if isinstance(transform, dict):
        return (
            "不可变清洗已完成，新数据集为 "
            f"`{transform.get('result_dataset_id', '—')}`。"
        )
    artifact = payload.get("artifact")
    if isinstance(artifact, dict):
        return (
            f"成果已生成：{artifact.get('artifact_type', 'artifact')} "
            f"`{artifact.get('artifact_id', '—')}`。"
        )
    return "数据分析工具执行完成，结果已记录。"


def _display_number(value: object) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4g}"
    return "—" if value is None else str(value)


def _route_tool(
    message: str, dataset_id: str, transcript: str
) -> tuple[str, dict[str, object]] | str:
    lowered = message.casefold()
    if re.search(r"导出|export", lowered):
        format_name = "json" if "json" in lowered else "csv"
        return (
            "create_analysis_report",
            {
                "dataset_id": dataset_id,
                "output_kind": "dataset_export",
                "analysis_ids": [],
                "plot_artifact_ids": [],
                "format": format_name,
            },
        )
    if re.search(
        r"生成报告|创建报告|导出报告|analysis\s+report|create\s+report",
        lowered,
    ):
        analysis_ids = tuple(dict.fromkeys(_ANALYSIS_ID.findall(transcript)))
        if not analysis_ids:
            return "请先执行统计分析，或明确提供 analysis_id。"
        return (
            "create_analysis_report",
            {
                "dataset_id": dataset_id,
                "output_kind": "analysis_report",
                "analysis_ids": analysis_ids,
                "plot_artifact_ids": tuple(
                    dict.fromkeys(_ARTIFACT_ID.findall(transcript))
                ),
                "format": "md",
            },
        )
    if re.search(r"图|plot|histogram|boxplot|scatter|heatmap", lowered):
        plot_type = _plot_type(lowered)
        columns = _csv_parameter(message, "columns")
        return (
            "create_analysis_plot",
            {
                "dataset_id": dataset_id,
                "plot_type": plot_type,
                "x": _parameter(message, "x"),
                "y": _parameter(message, "y"),
                "group_by": _parameter(message, "group_by"),
                "columns": columns,
                "correlation_method": (
                    "spearman" if "spearman" in lowered else "pearson"
                ),
            },
        )
    if re.search(r"清洗|去重|删除重复|transform|clean", lowered):
        subset = _csv_parameter(message, "subset")
        if not subset:
            return "请用 subset=字段名 指定去重字段。"
        return (
            "transform_dataset",
            {
                "dataset_id": dataset_id,
                "operations": [
                    {
                        "kind": "drop_duplicates",
                        "parameters": {"subset": subset, "keep": "first"},
                    }
                ],
            },
        )
    if re.search(r"检验|t检验|anova|mann|kruskal|test", lowered):
        response = _parameter(message, "response")
        group_by = _parameter(message, "group_by")
        groups = _csv_parameter(message, "groups")
        method = _test_method(lowered)
        if method == "paired_t":
            paired = _csv_parameter(message, "paired_columns")
            if len(paired) != 2:
                return "配对 t 检验请提供 paired_columns=字段1,字段2。"
            return (
                "run_statistical_test",
                {
                    "dataset_id": dataset_id,
                    "method": method,
                    "paired_columns": paired,
                },
            )
        if not response or not group_by or not groups:
            return "请提供 response=数值字段、group_by=分组字段和 groups=A,B。"
        return (
            "run_statistical_test",
            {
                "dataset_id": dataset_id,
                "method": method,
                "response_column": response,
                "group_column": group_by,
                "groups": groups,
            },
        )
    if re.search(r"相关|correlation|pearson|spearman", lowered):
        columns = _csv_parameter(message, "columns")
        if len(columns) < 2:
            return "请用 columns=字段1,字段2 指定至少两个数值字段。"
        return (
            "analyze_correlations",
            {
                "dataset_id": dataset_id,
                "columns": columns,
                "method": "spearman" if "spearman" in lowered else "pearson",
            },
        )
    if re.search(r"描述|统计|describe|summary", lowered):
        columns = _csv_parameter(message, "columns")
        if not columns:
            return "请用 columns=字段1,字段2 指定需要描述的数值字段。"
        return (
            "describe_dataset",
            {
                "dataset_id": dataset_id,
                "columns": columns,
                "group_by": _parameter(message, "group_by"),
            },
        )
    name = (
        "assess_data_quality"
        if re.search(r"质量|缺失|重复|quality", lowered)
        else "inspect_dataset"
    )
    return name, {"dataset_id": dataset_id}


def _parameter(message: str, name: str) -> str | None:
    match = re.search(
        rf"(?:^|\s|[，。；;]){re.escape(name)}=([^\s,，。；;]+)",
        message,
        re.IGNORECASE,
    )
    return match.group(1).strip() if match else None


def _csv_parameter(message: str, name: str) -> tuple[str, ...]:
    match = re.search(
        rf"(?:^|\s|[，。；;]){re.escape(name)}=([^\s。；;]+)",
        message,
        re.IGNORECASE,
    )
    if not match:
        return ()
    return tuple(
        value.strip()
        for value in re.split(r"[,，]", match.group(1))
        if value.strip()
    )


def _plot_type(message: str) -> str:
    for marker, value in (
        ("直方", "histogram"),
        ("histogram", "histogram"),
        ("箱线", "boxplot"),
        ("boxplot", "boxplot"),
        ("热力", "heatmap"),
        ("heatmap", "heatmap"),
        ("折线", "line"),
        ("line", "line"),
        ("柱", "bar"),
        ("bar", "bar"),
    ):
        if marker in message:
            return value
    return "scatter"


def _test_method(message: str) -> str:
    for marker, value in (
        ("welch", "welch_t"),
        ("paired", "paired_t"),
        ("配对", "paired_t"),
        ("mann", "mann_whitney"),
        ("anova", "anova"),
        ("kruskal", "kruskal_wallis"),
    ):
        if marker in message:
            return value
    return "student_t"


def _last_user_message(request: MaterialAgentRequest) -> str:
    for item in reversed(request.input_items):
        if isinstance(item, AgentMessageItem) and item.role == "user":
            return item.content
    return ""


def _all_text(request: MaterialAgentRequest) -> str:
    return "\n".join(
        item.content
        for item in request.input_items
        if isinstance(item, AgentMessageItem)
    )


def _message_response(content: Mapping[str, object]) -> MaterialAgentResponse:
    return MaterialAgentResponse(
        status=AgentModelStatus.COMPLETED,
        output_items=(
            AgentMessageItem(
                role="assistant", content=json.dumps(content, ensure_ascii=False)
            ),
        ),
        request_id="mock-data-analysis-message",
        provider="mock",
        model="mock-data-analysis",
    )
