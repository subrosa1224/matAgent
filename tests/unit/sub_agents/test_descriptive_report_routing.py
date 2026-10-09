"""A description report must reflect every current, source-linked tool row."""

import copy
import json
from pathlib import Path

import pytest
from tests.unit.agent_tools.helpers import FakeRunner, context

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
from materials_screening.data_analysis.dataset_store import DatasetStore
from materials_screening.data_analysis.statistics import DataStatisticsService
from materials_screening.sub_agents.data_analysis.models import DescribeDatasetInput
from materials_screening.sub_agents.data_analysis.routing_model import (
    DataAnalysisRoutingModel,
)
from materials_screening.sub_agents.data_analysis.tools import DescribeDatasetTool

MESSAGE = (
    "对 dataset-test 做描述统计。"
    "columns=band_gap_ev,density_g_cm3,energy_above_hull_ev_atom "
    "group_by=crystal_system。报告样本数、缺失值和分布，不推断器件性能。"
)
COLUMNS = ["band_gap_ev", "density_g_cm3", "energy_above_hull_ev_atom"]
GROUPS = [
    ("Cubic", 110),
    ("Hexagonal", 55),
    ("Monoclinic", 305),
    ("Orthorhombic", 257),
    ("Tetragonal", 100),
    ("Triclinic", 84),
    ("Trigonal", 89),
]


class _BadSummaryModel:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, request: MaterialAgentRequest) -> MaterialAgentResponse:
        del request
        self.calls += 1
        return MaterialAgentResponse(
            status=AgentModelStatus.COMPLETED,
            output_items=(
                AgentMessageItem(
                    role="assistant",
                    content=json.dumps(
                        {
                            "status": "completed",
                            "answer": "共6组，所有组中位数均为0",
                            "evidence_ids": ["evidence-description"],
                        },
                        ensure_ascii=False,
                    ),
                ),
            ),
            request_id="test",
            provider="test",
            model="test",
        )


def _payload() -> dict:
    statistics = []
    for group, count in GROUPS:
        for column in COLUMNS:
            median = (
                0.00022853462044203354
                if (group == "Triclinic" and column == "energy_above_hull_ev_atom")
                else 0.0
            )
            statistics.append(
                {
                    "column": column,
                    "group": group,
                    "count": count,
                    "missing_count": 0,
                    "mean": 2.0,
                    "std": 0.1,
                    "min": 0.0,
                    "max": 3.0,
                    "quantiles": {"0.25": 0.0, "0.5": median, "0.75": 2.5},
                }
            )
    return {
        "evidence_id": "evidence-description",
        "dataset": {
            "dataset_id": "dataset-test",
            "source_artifact_id": "artifact-test",
            "display_name": "test.json",
            "format": "json",
            "row_count": 1000,
            "column_count": 4,
            "schema_fingerprint": "sha256:" + "0" * 64,
            "content_fingerprint": "sha256:" + "1" * 64,
        },
        "analysis": {
            "analysis_id": "analysis-test",
            "dataset_id": "dataset-test",
            "analysis_type": "descriptive",
            "method": "describe",
            "parameters": {
                "columns": COLUMNS,
                "group_by": "crystal_system",
                "quantiles": [0.25, 0.5, 0.75],
                "missing_strategy": "drop_per_column",
            },
            "summary": {"statistics": statistics},
            "warnings": [],
            "evidence_id": "evidence-description",
        },
    }


def _request(
    payload: dict, *, message: str = MESSAGE, status: str = "ok"
) -> MaterialAgentRequest:
    envelope = {
        "status": status,
        "tool_name": "describe_dataset",
        "call_id": "describe",
        "evidence_id": "evidence-description",
        "output": payload,
    }
    if status == "error":
        envelope.update(
            output=None,
            error={
                "code": "TOOL_FAILED",
                "message": "unknown column",
                "retryable": False,
            },
        )
    return MaterialAgentRequest(
        instructions="analysis",
        tool_definitions=(),
        final_draft_schema={},
        input_items=(
            AgentMessageItem(role="user", content=message),
            AgentFunctionCallItem(
                call_id="describe",
                name="describe_dataset",
                arguments=json.dumps(
                    {
                        "dataset_id": "dataset-test",
                        "columns": COLUMNS,
                        "group_by": "crystal_system",
                    }
                ),
            ),
            AgentFunctionOutputItem(call_id="describe", output=json.dumps(envelope)),
        ),
    )


def test_report_keeps_all_seven_groups_and_nonzero_median_without_model() -> None:
    delegate = _BadSummaryModel()
    response = DataAnalysisRoutingModel(delegate).generate(_request(_payload()))
    draft = json.loads(response.message_text)
    assert draft["status"] == "completed"
    assert "分组数：7" in draft["answer"]
    assert "样本数：1000" in draft["answer"]
    assert "Trigonal" in draft["answer"] and "89" in draft["answer"]
    assert "0.00022853462" in draft["answer"]
    assert "所有组中位数均为0" not in draft["answer"]
    assert draft["evidence_ids"] == ["evidence-description"]
    assert delegate.calls == 0


@pytest.mark.parametrize(
    "damage", ["lost_group", "duplicate", "negative", "unequal_rows"]
)
def test_inconsistent_counts_fail_closed_instead_of_becoming_prose(damage: str) -> None:
    payload = _payload()
    rows = payload["analysis"]["summary"]["statistics"]
    if damage == "lost_group":
        del rows[-3:]
    elif damage == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif damage == "negative":
        rows[0]["count"] = -1
    else:
        rows[0]["count"] += 1
    delegate = _BadSummaryModel()
    draft = json.loads(
        DataAnalysisRoutingModel(delegate).generate(_request(payload)).message_text
    )
    assert draft["status"] == "error"
    assert "统计结果一致性校验失败" in draft["answer"]
    assert delegate.calls == 0


def test_failed_tool_is_not_a_successful_description() -> None:
    delegate = _BadSummaryModel()
    response = DataAnalysisRoutingModel(delegate).generate(
        _request(_payload(), status="error")
    )
    draft = json.loads(response.message_text)
    assert draft["status"] == "error"
    assert "unknown column" in draft["answer"]
    assert "分组数：7" not in draft["answer"]
    assert delegate.calls == 0


@pytest.mark.parametrize(
    "message",
    [
        MESSAGE + "另做Pearson相关分析。",
        MESSAGE + "并生成散点图。",
        MESSAGE + "导出CSV。",
        MESSAGE.replace("dataset-test", "dataset-other"),
    ],
)
def test_other_tasks_or_datasets_are_not_prematurely_finalized(message: str) -> None:
    delegate = _BadSummaryModel()
    DataAnalysisRoutingModel(delegate).generate(_request(_payload(), message=message))
    assert delegate.calls == 1


def test_old_turn_describe_output_is_not_reused_for_current_turn() -> None:
    request = _request(_payload())
    request = MaterialAgentRequest(
        instructions=request.instructions,
        tool_definitions=(),
        final_draft_schema={},
        input_items=(
            *request.input_items,
            AgentMessageItem(role="user", content=MESSAGE),
        ),
    )
    delegate = _BadSummaryModel()
    response = DataAnalysisRoutingModel(delegate).generate(request)
    assert response.tool_calls[0].name == "assess_data_quality"
    assert delegate.calls == 0


def test_real_describe_tool_includes_row_count_for_consistency_validation(
    tmp_path: Path,
) -> None:
    source = tmp_path / "data.csv"
    source.write_text("group,x\nA,1\nA,\nB,3\n", encoding="utf-8")
    store = DatasetStore(tmp_path / "store")
    dataset = store.register_file(source, source_artifact_id="artifact-real-test")
    payload = DescribeDatasetTool(DataStatisticsService(store)).execute(
        DescribeDatasetInput(
            dataset_id=dataset.dataset_id, columns=("x",), group_by="group"
        ),
        context(runner=FakeRunner(), call_id="description"),
    )
    assert payload.dataset == dataset
    assert payload.dataset.row_count == 3


def test_missing_values_are_not_counted_as_extra_rows() -> None:
    payload = _payload()
    rows = payload["analysis"]["summary"]["statistics"]
    rows[0].update(count=108, missing_count=2)
    delegate = _BadSummaryModel()
    draft = json.loads(
        DataAnalysisRoutingModel(delegate).generate(_request(payload)).message_text
    )
    assert draft["status"] == "completed"
    assert "样本数：1000" in draft["answer"]
    assert "band_gap_ev：有效=998，缺失/非有限=2" in draft["answer"]
    assert "| Cubic | 110 |" in draft["answer"]
    assert delegate.calls == 0


@pytest.mark.parametrize(
    "damage", ["evidence", "dataset", "columns", "quantiles", "null_value"]
)
def test_corrupted_metadata_or_statistics_cannot_be_reported_as_success(
    damage: str,
) -> None:
    payload = _payload()
    if damage == "evidence":
        payload["analysis"]["evidence_id"] = "evidence-other"
    elif damage == "dataset":
        payload["dataset"]["dataset_id"] = "dataset-other"
    elif damage == "columns":
        payload["analysis"]["parameters"]["columns"] = ["density_g_cm3"]
    elif damage == "quantiles":
        payload["analysis"]["summary"]["statistics"][0]["quantiles"].pop("0.5")
    else:
        payload["analysis"]["summary"]["statistics"][0]["mean"] = None
    delegate = _BadSummaryModel()
    draft = json.loads(
        DataAnalysisRoutingModel(delegate).generate(_request(payload)).message_text
    )
    assert draft["status"] == "error"
    assert delegate.calls == 0


def test_many_tool_rows_are_bounded_without_claiming_full_display() -> None:
    payload = _payload()
    rows = payload["analysis"]["summary"]["statistics"]
    template = rows[:3]
    rows[:] = [
        dict(row, group=f"group-{index}", count=10)
        for index in range(100)
        for row in template
    ]
    draft = json.loads(
        DataAnalysisRoutingModel(_BadSummaryModel())
        .generate(_request(payload))
        .message_text
    )
    assert draft["status"] == "completed"
    assert "分组数：100" in draft["answer"]
    assert len(draft["answer"]) <= 8000
    assert any("/300" in warning for warning in draft["warnings"])


def test_overall_statistics_label_is_not_a_missing_group() -> None:
    payload = _payload()
    payload["analysis"]["parameters"]["group_by"] = None
    rows = payload["analysis"]["summary"]["statistics"]
    rows[:] = [dict(row, group=None, count=1000) for row in rows[:3]]
    request = _request(payload, message=MESSAGE.replace("group_by=crystal_system", ""))
    call = request.input_items[1].model_copy(
        update={
            "arguments": json.dumps(
                {
                    "dataset_id": "dataset-test",
                    "columns": COLUMNS,
                }
            )
        }
    )
    request = MaterialAgentRequest(
        instructions="analysis",
        tool_definitions=(),
        final_draft_schema={},
        input_items=(request.input_items[0], call, request.input_items[2]),
    )
    draft = json.loads(
        DataAnalysisRoutingModel(_BadSummaryModel()).generate(request).message_text
    )
    assert draft["status"] == "completed"
    assert "未分组" in draft["answer"]
    assert "缺失分组" not in draft["answer"]


def test_graph_executes_real_tools_and_validates_the_direct_report(
    tmp_path: Path,
) -> None:
    from langgraph.checkpoint.memory import InMemorySaver

    from materials_screening.agent.conversation_store import SqliteConversationStore
    from materials_screening.agent.runner import MaterialAgentRunner
    from materials_screening.agent.settings import AgentSettings
    from materials_screening.agent_tools.result_reader import FileWorkflowResultReader
    from materials_screening.data_analysis.reporting import DataAnalysisReportingService
    from materials_screening.data_analysis.service import DataAnalysisService
    from materials_screening.data_analysis.transform import DatasetTransformService
    from materials_screening.sub_agents.data_analysis.tools import build_tool_registry

    source = tmp_path / "groups.csv"
    source.write_text(
        "crystal_system,band_gap_ev,density_g_cm3,energy_above_hull_ev_atom\n"
        + "".join(
            f"{group},2,5,0.00022853462044203354\n"
            for group, count in GROUPS
            for _ in range(count)
        ),
        encoding="utf-8",
    )
    store = DatasetStore(tmp_path / "datasets")
    dataset = store.register_file(source, source_artifact_id="artifact-graph-fixture")
    registry = build_tool_registry(
        DataAnalysisService(store),
        DataStatisticsService(store),
        DatasetTransformService(store),
        DataAnalysisReportingService(store),
    )

    class _ToolSelectionOnly:
        calls = 0

        def generate(self, request: MaterialAgentRequest) -> MaterialAgentResponse:
            del request
            self.calls += 1
            assert self.calls == 1, "No model-generated numerical prose allowed"
            return MaterialAgentResponse(
                status=AgentModelStatus.COMPLETED,
                output_items=(
                    AgentFunctionCallItem(
                        call_id="description",
                        name="describe_dataset",
                        arguments=json.dumps(
                            {
                                "dataset_id": dataset.dataset_id,
                                "columns": COLUMNS,
                                "group_by": "crystal_system",
                            }
                        ),
                    ),
                ),
                request_id="select-only",
                provider="test",
                model="test",
            )

    delegate = _ToolSelectionOnly()
    runner = MaterialAgentRunner(
        settings=AgentSettings(
            _env_file=None,
            agent_allow_multi_step_tools=True,
            agent_max_model_calls_per_turn=6,
            agent_max_tool_calls_per_turn=5,
        ),
        store=SqliteConversationStore(tmp_path / "conversations.sqlite"),
        workflow_runner=FakeRunner(),
        workflow_result_reader=FileWorkflowResultReader(tmp_path / "runs"),
        tool_registry=registry,
        agent_model=DataAnalysisRoutingModel(delegate),
        checkpointer=InMemorySaver(),
    )
    result = runner.ask(message=MESSAGE.replace("dataset-test", dataset.dataset_id))
    assert result.status == result.final_status == "completed"
    assert result.selected_tools == ("assess_data_quality", "describe_dataset")
    assert result.tool_call_count == 2
    assert (
        "分组数：7" in result.response_text and "样本数：1000" in result.response_text
    )
    assert (
        "Trigonal" in result.response_text and "0.00022853462" in result.response_text
    )
    assert delegate.calls == 0
