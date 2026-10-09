from __future__ import annotations

import json
from pathlib import Path

import pytest
from tests.unit.agent_tools.helpers import FakeRunner, context

from materials_screening.agent.model_base import (
    AgentModelStatus,
    MaterialAgentRequest,
    MaterialAgentResponse,
)
from materials_screening.agent.models import AgentFunctionCallItem, AgentMessageItem
from materials_screening.agent_tools.result_reader import FileWorkflowResultReader
from materials_screening.data_analysis.dataset_store import DatasetStore
from materials_screening.data_analysis.reporting import DataAnalysisReportingService
from materials_screening.data_analysis.service import DataAnalysisService
from materials_screening.data_analysis.statistics import DataStatisticsService
from materials_screening.data_analysis.transform import DatasetTransformService
from materials_screening.sub_agents.data_analysis.mock_model import (
    DataAnalysisMockModel,
    _summarize_payload,
)
from materials_screening.sub_agents.data_analysis.models import (
    AnalyzeCorrelationsInput,
    AssessDataQualityInput,
    CreateAnalysisPlotInput,
    CreateAnalysisReportInput,
    DescribeDatasetInput,
    InspectDatasetInput,
    RunStatisticalTestInput,
    TransformDatasetInput,
)
from materials_screening.sub_agents.data_analysis.routing_model import (
    DataAnalysisRoutingModel,
)
from materials_screening.sub_agents.data_analysis.spec_factory import create_spec
from materials_screening.sub_agents.data_analysis.tools import build_tool_registry


def _registry(tmp_path: Path) -> tuple[object, DatasetStore, str]:
    source = tmp_path / "agent.csv"
    source.write_text(
        "group,x,y\nA,1,2\nA,2,4\nA,3,6\nB,4,8\nB,5,10\nB,6,12\n",
        encoding="utf-8",
    )
    store = DatasetStore(tmp_path / "private")
    dataset = store.register_file(
        source, source_artifact_id="artifact-data-agent-test"
    )
    registry = build_tool_registry(
        DataAnalysisService(store),
        DataStatisticsService(store),
        DatasetTransformService(store),
        DataAnalysisReportingService(store),
    )
    return registry, store, dataset.dataset_id


def test_registry_exposes_exactly_eight_strict_whitelisted_tools(
    tmp_path: Path,
) -> None:
    registry, _, _ = _registry(tmp_path)

    assert registry.names() == (
        "analyze_correlations",
        "assess_data_quality",
        "create_analysis_plot",
        "create_analysis_report",
        "describe_dataset",
        "inspect_dataset",
        "run_statistical_test",
        "transform_dataset",
    )
    assert all(
        definition.parameters["additionalProperties"] is False
        for definition in registry.definitions()
    )


def test_read_only_tools_record_evidence_and_structured_results(
    tmp_path: Path,
) -> None:
    registry, _, dataset_id = _registry(tmp_path)
    cases = [
        ("inspect_dataset", InspectDatasetInput(dataset_id=dataset_id)),
        ("assess_data_quality", AssessDataQualityInput(dataset_id=dataset_id)),
        (
            "describe_dataset",
            DescribeDatasetInput(dataset_id=dataset_id, columns=("x", "y")),
        ),
        (
            "analyze_correlations",
            AnalyzeCorrelationsInput(dataset_id=dataset_id, columns=("x", "y")),
        ),
        (
            "run_statistical_test",
            RunStatisticalTestInput(
                dataset_id=dataset_id,
                method="welch_t",
                response_column="x",
                group_column="group",
                groups=("A", "B"),
            ),
        ),
    ]
    for index, (name, arguments) in enumerate(cases, 1):
        tool_context = context(runner=FakeRunner(), call_id=f"call-{index}")
        result = registry.get(name).execute(arguments, tool_context)
        assert result.evidence_id.startswith("evidence-")  # type: ignore[attr-defined]
        assert tool_context.ledger.executed_tool_names() == (name,)
        assert not tool_context.ledger.side_effect_executed()


def test_mutating_and_artifact_tools_are_explicit_side_effects(tmp_path: Path) -> None:
    registry, _, dataset_id = _registry(tmp_path)
    transform_context = context(runner=FakeRunner(), call_id="transform")
    transformed = registry.get("transform_dataset").execute(
        TransformDatasetInput(
            dataset_id=dataset_id,
            operations=(
                {"kind": "select_columns", "parameters": {"columns": ["x", "y"]}},
            ),
        ),
        transform_context,
    )
    assert transformed.dataset.parent_dataset_id == dataset_id  # type: ignore[attr-defined,union-attr]
    assert transform_context.ledger.side_effect_executed()

    plot_context = context(runner=FakeRunner(), call_id="plot")
    plot = registry.get("create_analysis_plot").execute(
        CreateAnalysisPlotInput(
            dataset_id=dataset_id, plot_type="scatter", x="x", y="y"
        ),
        plot_context,
    )
    assert plot.artifact.artifact_type == "analysis_plot"  # type: ignore[attr-defined,union-attr]

    describe_context = context(runner=FakeRunner(), call_id="describe")
    described = registry.get("describe_dataset").execute(
        DescribeDatasetInput(dataset_id=dataset_id, columns=("x",)),
        describe_context,
    )
    report_context = context(runner=FakeRunner(), call_id="report")
    report = registry.get("create_analysis_report").execute(
        CreateAnalysisReportInput(
            dataset_id=dataset_id,
            analysis_ids=(described.analysis.analysis_id,),  # type: ignore[attr-defined,union-attr]
            plot_artifact_ids=(plot.artifact.artifact_id,),  # type: ignore[attr-defined,union-attr]
            format="md",
        ),
        report_context,
    )
    assert report.artifact.artifact_type == "analysis_report"  # type: ignore[attr-defined,union-attr]
    assert report_context.ledger.side_effect_executed()


class _Delegate:
    def generate(self, request: MaterialAgentRequest) -> MaterialAgentResponse:
        del request
        return MaterialAgentResponse(
            status=AgentModelStatus.COMPLETED,
            output_items=(AgentMessageItem(role="assistant", content="delegate"),),
            request_id="delegate",
            provider="test",
            model="test",
        )


def _request(message: str) -> MaterialAgentRequest:
    return MaterialAgentRequest(
        input_items=(AgentMessageItem(role="user", content=message),),
        instructions="data analysis",
        tool_definitions=(),
        final_draft_schema={"type": "object"},
    )


def test_deterministic_router_only_intercepts_complete_inspection_requests() -> None:
    model = DataAnalysisRoutingModel(_Delegate())
    response = model.generate(_request("检查 dataset-source-1 的字段和预览"))
    assert isinstance(response.output_items[0], AgentFunctionCallItem)
    assert response.output_items[0].name == "inspect_dataset"

    quality = model.generate(_request("检查 dataset-source-1 的数据质量和缺失值"))
    assert quality.output_items[0].name == "assess_data_quality"  # type: ignore[attr-defined]

    delegated = model.generate(_request("帮我分析一下数据"))
    assert delegated.message_text == "delegate"


def test_mock_model_requests_dataset_id_and_routes_offline_inspection() -> None:
    model = DataAnalysisMockModel()
    clarification = model.generate(_request("检查数据质量"))
    draft = json.loads(clarification.message_text)
    assert draft["status"] == "needs_user_input"

    routed = model.generate(_request("检查 dataset-source-1 的数据质量"))
    assert routed.tool_calls[0].name == "assess_data_quality"


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (
            "描述 dataset-source-1 columns=x,y group_by=group",
            "describe_dataset",
        ),
        (
            "描述 dataset-source-1 columns=x,y group_by=group，只报告观察性结果",
            "describe_dataset",
        ),
        (
            "对 dataset-source-1 做描述统计。"
            "columns=x,y group_by=group。仅输出观察结果",
            "describe_dataset",
        ),
        (
            "Pearson相关 dataset-source-1 columns=x,y",
            "analyze_correlations",
        ),
        (
            "Welch检验 dataset-source-1 response=x group_by=group groups=A,B",
            "run_statistical_test",
        ),
        (
            "清洗 dataset-source-1 subset=group",
            "transform_dataset",
        ),
        (
            "散点图 dataset-source-1 x=x y=y group_by=group",
            "create_analysis_plot",
        ),
        (
            "生成报告 dataset-source-1 analysis-source-1",
            "create_analysis_report",
        ),
        (
            "导出 dataset-source-1 JSON",
            "create_analysis_report",
        ),
    ],
)
def test_mock_model_routes_every_explicit_analysis_intent(
    message: str, expected: str
) -> None:
    response = DataAnalysisMockModel().generate(_request(message))

    assert response.tool_calls[0].name == expected


def test_mock_model_surfaces_grouped_descriptive_statistics() -> None:
    text = _summarize_payload(
        {
            "analysis": {
                "analysis_type": "descriptive",
                "analysis_id": "analysis-test",
                "summary": {
                    "statistics": [
                        {
                            "column": "numeric_value",
                            "group": "length [nm]",
                            "count": 2,
                            "mean": 90.0,
                            "min": 60.0,
                            "max": 120.0,
                        }
                    ]
                },
            }
        }
    )

    assert "analysis-test" in text
    assert "length [nm]：n=2" in text
    assert "范围=60–120" in text


def test_mock_model_keeps_property_name_for_database_groups() -> None:
    text = _summarize_payload(
        {
            "analysis": {
                "analysis_type": "descriptive",
                "analysis_id": "analysis-materials",
                "summary": {
                    "statistics": [
                        {
                            "column": "density_g_cm3",
                            "group": "Cubic",
                            "count": 3,
                            "mean": 5.2,
                            "min": 4.1,
                            "max": 6.3,
                        }
                    ]
                },
            }
        }
    )

    assert "density_g_cm3 [Cubic]：n=3" in text


def test_spec_factory_declares_data_analysis_without_network(tmp_path: Path) -> None:
    spec = create_spec(
        workflow_runner=FakeRunner(),
        result_reader=FileWorkflowResultReader(tmp_path / "runs"),
        data_root=tmp_path / "data-analysis",
    )

    assert spec.name == "data_analysis"
    assert spec.delegate_function_name == "delegate_to_data_analysis"
    assert len(spec.tool_definitions) == 8
    assert "arbitrary code" not in spec.description.lower()
