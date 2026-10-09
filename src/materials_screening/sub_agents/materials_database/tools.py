"""Whitelisted tools for the unified Materials Project database agent."""

from __future__ import annotations

import uuid
from typing import Any

from materials_screening.agent.context import AgentToolContext
from materials_screening.agent.tool_base import ToolSideEffect
from materials_screening.agent.tool_registry import AgentToolRegistry
from materials_screening.services.material_database_service import (
    MaterialDatabaseService,
)
from materials_screening.services.outlier_detection_service import (
    detect_multivariate_outliers,
    detect_property_outliers,
)

from .models import (
    CompareMaterialsInput,
    DescribeMaterialsInput,
    DetectOutliersInput,
    ExportMaterialsInput,
    GetMaterialDetailsInput,
    GetQueryResultInput,
    SearchMaterialsInput,
    ToolPayload,
)


class _DatabaseTool:
    side_effect = ToolSideEffect.READ_ONLY
    output_model = ToolPayload

    def __init__(self, service: MaterialDatabaseService) -> None:
        self.service = service

    def _record(
        self, payload: dict[str, Any], context: AgentToolContext
    ) -> ToolPayload:
        evidence_id = context.id_generator.new_id()
        payload["evidence_id"] = evidence_id
        result = ToolPayload.model_validate(payload)
        context.ledger.record(
            call_id=context.call_id,
            tool_name=self.name,
            evidence_id=evidence_id,
            result_json=result.model_dump_json(),
            side_effect=self.side_effect,
        )
        return result


class SearchMaterialsTool(_DatabaseTool):
    name = "search_materials"
    description = (
        "Query Materials Project with structured filters, sorting and Top-K. "
        "Only include fields requested by the user or needed by the operation."
    )
    input_model = SearchMaterialsInput

    def execute(
        self, arguments: SearchMaterialsInput, context: AgentToolContext
    ) -> ToolPayload:
        return self._record(self.service.search(arguments), context)


class GetMaterialDetailsTool(_DatabaseTool):
    name = "get_material_details"
    description = (
        "Get selected details for up to 20 explicit Materials Project material IDs."
    )
    input_model = GetMaterialDetailsInput

    def execute(
        self, arguments: GetMaterialDetailsInput, context: AgentToolContext
    ) -> ToolPayload:
        return self._record(
            self.service.details(arguments.material_ids, arguments.fields), context
        )


class GetQueryResultTool(_DatabaseTool):
    name = "get_query_result"
    description = (
        "Read a page from a previous query snapshot without querying "
        "Materials Project again."
    )
    input_model = GetQueryResultInput

    def execute(
        self, arguments: GetQueryResultInput, context: AgentToolContext
    ) -> ToolPayload:
        return self._record(
            self.service.page(
                arguments.query_id, arguments.offset, arguments.limit, arguments.fields
            ),
            context,
        )


class CompareMaterialsTool(_DatabaseTool):
    name = "compare_materials"
    description = (
        "Compare explicitly requested numeric properties for materials "
        "or a prior query result."
    )
    input_model = CompareMaterialsInput

    def execute(
        self, arguments: CompareMaterialsInput, context: AgentToolContext
    ) -> ToolPayload:
        return self._record(
            self.service.compare(
                arguments.material_ids, arguments.query_id, arguments.properties
            ),
            context,
        )


class DescribeMaterialsTool(_DatabaseTool):
    name = "describe_materials"
    description = (
        "Calculate descriptive statistics for explicitly requested numeric "
        "properties in a query snapshot."
    )
    input_model = DescribeMaterialsInput

    def execute(
        self, arguments: DescribeMaterialsInput, context: AgentToolContext
    ) -> ToolPayload:
        return self._record(
            self.service.describe(
                arguments.query_id, arguments.properties, arguments.include_correlation
            ),
            context,
        )


class DetectMaterialOutliersTool(_DatabaseTool):
    name = "detect_material_outliers"
    description = (
        "Detect outliers only when the user explicitly requests anomaly or "
        "outlier analysis, using a prior query snapshot."
    )
    input_model = DetectOutliersInput

    def execute(
        self, arguments: DetectOutliersInput, context: AgentToolContext
    ) -> ToolPayload:
        records = self.service.store.load_records(arguments.query_id)
        if len(arguments.properties) == 1:
            method = (
                arguments.method if arguments.method in {"zscore", "iqr"} else "iqr"
            )
            report = detect_property_outliers(
                records, arguments.properties[0], method, arguments.threshold
            )
        else:
            method = (
                arguments.method
                if arguments.method in {"mahalanobis", "isolation_forest"}
                else "mahalanobis"
            )
            report = detect_multivariate_outliers(records, arguments.properties, method)
        analysis_id = f"analysis-{uuid.uuid4().hex}"
        payload = {
            "analysis_id": analysis_id,
            "query_id": arguments.query_id,
            "analysis_type": "outlier",
            "report": report.model_dump(mode="json"),
        }
        self.service.store.save_analysis(analysis_id, payload)
        return self._record(payload, context)


class ExportMaterialsTool(_DatabaseTool):
    name = "export_materials"
    description = (
        "Export a query or analysis result only when the user explicitly asks "
        "for a CSV, JSON or Markdown file."
    )
    input_model = ExportMaterialsInput
    side_effect = ToolSideEffect.CREATE_WORKFLOW_RUN

    def execute(
        self, arguments: ExportMaterialsInput, context: AgentToolContext
    ) -> ToolPayload:
        return self._record(
            self.service.export(
                arguments.query_id,
                arguments.analysis_id,
                arguments.format,
                arguments.fields,
            ),
            context,
        )


def build_tool_registry(service: MaterialDatabaseService) -> AgentToolRegistry:
    return AgentToolRegistry(
        (
            SearchMaterialsTool(service),
            GetMaterialDetailsTool(service),
            GetQueryResultTool(service),
            CompareMaterialsTool(service),
            DescribeMaterialsTool(service),
            DetectMaterialOutliersTool(service),
            ExportMaterialsTool(service),
        )
    )
