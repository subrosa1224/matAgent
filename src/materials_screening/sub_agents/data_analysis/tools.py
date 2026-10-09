"""Eight whitelisted tools for deterministic data analysis."""

from __future__ import annotations

import re
import uuid
from typing import Literal

from materials_screening.agent.context import AgentToolContext
from materials_screening.agent.tool_base import ToolSideEffect
from materials_screening.agent.tool_registry import AgentToolRegistry
from materials_screening.data_analysis.reporting import DataAnalysisReportingService
from materials_screening.data_analysis.service import DataAnalysisService
from materials_screening.data_analysis.statistics import DataStatisticsService
from materials_screening.data_analysis.transform import DatasetTransformService

from .models import (
    AnalyzeCorrelationsInput,
    AssessDataQualityInput,
    CreateAnalysisPlotInput,
    CreateAnalysisReportInput,
    DataAnalysisToolPayload,
    DescribeDatasetInput,
    InspectDatasetInput,
    RunStatisticalTestInput,
    TransformDatasetInput,
)


class _DataAnalysisTool:
    name: str
    description: str
    side_effect = ToolSideEffect.READ_ONLY
    output_model = DataAnalysisToolPayload

    def _record(
        self,
        payload: DataAnalysisToolPayload,
        context: AgentToolContext,
    ) -> DataAnalysisToolPayload:
        context.ledger.record(
            call_id=context.call_id,
            tool_name=self.name,
            evidence_id=payload.evidence_id,
            result_json=payload.model_dump_json(),
            side_effect=self.side_effect,
        )
        return payload


class InspectDatasetTool(_DataAnalysisTool):
    name = "inspect_dataset"
    description = "Inspect schema, a bounded preview, and basic quality signals."
    input_model = InspectDatasetInput

    def __init__(self, service: DataAnalysisService) -> None:
        self.service = service

    def execute(
        self, arguments: InspectDatasetInput, context: AgentToolContext
    ) -> DataAnalysisToolPayload:
        evidence = _evidence_id(context)
        inspection = self.service.inspect_dataset(
            arguments.dataset_id, offset=arguments.offset, limit=arguments.limit
        )
        return self._record(
            DataAnalysisToolPayload(
                evidence_id=evidence,
                dataset=inspection.dataset,
                inspection=inspection,
            ),
            context,
        )


class AssessDataQualityTool(_DataAnalysisTool):
    name = "assess_data_quality"
    description = "Assess missing values, duplicates, mixed types and IQR warnings."
    input_model = AssessDataQualityInput

    def __init__(self, service: DataAnalysisService) -> None:
        self.service = service

    def execute(
        self, arguments: AssessDataQualityInput, context: AgentToolContext
    ) -> DataAnalysisToolPayload:
        evidence = _evidence_id(context)
        inspection = self.service.inspect_dataset(arguments.dataset_id, limit=1)
        return self._record(
            DataAnalysisToolPayload(
                evidence_id=evidence,
                dataset=inspection.dataset,
                inspection=inspection,
                metadata={"preview_is_diagnostic_only": True},
            ),
            context,
        )


class DescribeDatasetTool(_DataAnalysisTool):
    name = "describe_dataset"
    description = "Calculate explicit overall or grouped descriptive statistics."
    input_model = DescribeDatasetInput

    def __init__(self, service: DataStatisticsService) -> None:
        self.service = service

    def execute(
        self, arguments: DescribeDatasetInput, context: AgentToolContext
    ) -> DataAnalysisToolPayload:
        evidence = _evidence_id(context)
        result = self.service.describe_dataset(
            arguments.dataset_id,
            columns=arguments.columns,
            group_by=arguments.group_by,
            quantiles=arguments.quantiles,
            evidence_id=evidence,
        )
        return self._record(
            DataAnalysisToolPayload(
                evidence_id=evidence,
                dataset=self.service.store.get(arguments.dataset_id),
                analysis=result,
            ),
            context,
        )


class AnalyzeCorrelationsTool(_DataAnalysisTool):
    name = "analyze_correlations"
    description = "Calculate Pearson or Spearman pairwise correlations."
    input_model = AnalyzeCorrelationsInput

    def __init__(self, service: DataStatisticsService) -> None:
        self.service = service

    def execute(
        self, arguments: AnalyzeCorrelationsInput, context: AgentToolContext
    ) -> DataAnalysisToolPayload:
        evidence = _evidence_id(context)
        result = self.service.analyze_correlations(
            arguments.dataset_id,
            columns=arguments.columns,
            method=arguments.method,
            evidence_id=evidence,
        )
        return self._record(
            DataAnalysisToolPayload(evidence_id=evidence, analysis=result), context
        )


class RunStatisticalTestTool(_DataAnalysisTool):
    name = "run_statistical_test"
    description = "Run one explicitly selected supported statistical test."
    input_model = RunStatisticalTestInput

    def __init__(self, service: DataStatisticsService) -> None:
        self.service = service

    def execute(
        self, arguments: RunStatisticalTestInput, context: AgentToolContext
    ) -> DataAnalysisToolPayload:
        evidence = _evidence_id(context)
        result = self.service.run_statistical_test(
            arguments.dataset_id,
            method=arguments.method,
            response_column=arguments.response_column,
            group_column=arguments.group_column,
            groups=arguments.groups,
            paired_columns=arguments.paired_columns,
            alpha=arguments.alpha,
            evidence_id=evidence,
        )
        return self._record(
            DataAnalysisToolPayload(evidence_id=evidence, analysis=result), context
        )


class TransformDatasetTool(_DataAnalysisTool):
    name = "transform_dataset"
    description = "Create a new immutable dataset using explicit cleaning operations."
    input_model = TransformDatasetInput
    side_effect = ToolSideEffect.CREATE_WORKFLOW_RUN

    def __init__(self, service: DatasetTransformService) -> None:
        self.service = service

    def execute(
        self, arguments: TransformDatasetInput, context: AgentToolContext
    ) -> DataAnalysisToolPayload:
        evidence = _evidence_id(context)
        dataset, record, inspection = self.service.transform_dataset(
            arguments.dataset_id,
            operations=arguments.operations,
        )
        return self._record(
            DataAnalysisToolPayload(
                evidence_id=evidence,
                dataset=dataset,
                transform=record,
                inspection=inspection,
            ),
            context,
        )


class CreateAnalysisPlotTool(_DataAnalysisTool):
    name = "create_analysis_plot"
    description = "Create one fixed, bounded PNG chart."
    input_model = CreateAnalysisPlotInput
    side_effect = ToolSideEffect.CREATE_WORKFLOW_RUN

    def __init__(self, service: DataAnalysisReportingService) -> None:
        self.service = service

    def execute(
        self, arguments: CreateAnalysisPlotInput, context: AgentToolContext
    ) -> DataAnalysisToolPayload:
        evidence = _evidence_id(context)
        artifact = self.service.create_plot(
            arguments.dataset_id,
            plot_type=arguments.plot_type,
            x=arguments.x,
            y=arguments.y,
            group_by=arguments.group_by,
            columns=arguments.columns,
            title=arguments.title,
            x_label=arguments.x_label,
            y_label=arguments.y_label,
            correlation_method=arguments.correlation_method,
        )
        return self._record(
            DataAnalysisToolPayload(evidence_id=evidence, artifact=artifact), context
        )


class CreateAnalysisReportTool(_DataAnalysisTool):
    name = "create_analysis_report"
    description = "Create an explicit report or safe dataset export artifact."
    input_model = CreateAnalysisReportInput
    side_effect = ToolSideEffect.CREATE_WORKFLOW_RUN

    def __init__(self, service: DataAnalysisReportingService) -> None:
        self.service = service

    def execute(
        self, arguments: CreateAnalysisReportInput, context: AgentToolContext
    ) -> DataAnalysisToolPayload:
        evidence = _evidence_id(context)
        if arguments.output_kind == "analysis_report":
            artifact = self.service.create_report(
                arguments.dataset_id,
                analysis_ids=arguments.analysis_ids,
                plot_artifact_ids=arguments.plot_artifact_ids,
                format=arguments.format,
            )
        else:
            export_format: Literal["csv", "json"] = (
                "csv" if arguments.format == "csv" else "json"
            )
            artifact = self.service.export_dataset(
                arguments.dataset_id, format=export_format
            )
        return self._record(
            DataAnalysisToolPayload(evidence_id=evidence, artifact=artifact), context
        )


def build_tool_registry(
    inspection: DataAnalysisService,
    statistics: DataStatisticsService,
    transform: DatasetTransformService,
    reporting: DataAnalysisReportingService,
) -> AgentToolRegistry:
    tools = (
        InspectDatasetTool(inspection),
        AssessDataQualityTool(inspection),
        DescribeDatasetTool(statistics),
        AnalyzeCorrelationsTool(statistics),
        RunStatisticalTestTool(statistics),
        TransformDatasetTool(transform),
        CreateAnalysisPlotTool(reporting),
        CreateAnalysisReportTool(reporting),
    )
    return AgentToolRegistry(tools)  # type: ignore[arg-type]


def _evidence_id(context: AgentToolContext) -> str:
    raw = context.id_generator.new_id()
    safe = re.sub(r"[^A-Za-z0-9._:-]+", "-", raw).strip("-._:")
    return f"evidence-{safe or uuid.uuid4().hex}"
