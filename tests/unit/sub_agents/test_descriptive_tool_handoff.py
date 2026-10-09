"""Explicit statistics must reach the real tool even when the model is offline."""

import json

import pytest
from tests.unit.sub_agents.test_descriptive_report_routing import (
    COLUMNS,
    MESSAGE,
    _BadSummaryModel,
    _payload,
)

from materials_screening.agent.errors import AgentModelError
from materials_screening.agent.model_base import MaterialAgentRequest
from materials_screening.agent.models import (
    AgentFunctionCallItem,
    AgentFunctionOutputItem,
    AgentMessageItem,
)
from materials_screening.sub_agents.data_analysis.routing_model import (
    DataAnalysisRoutingModel,
)


class _OfflineModel:
    calls = 0

    def generate(self, request: MaterialAgentRequest):
        del request
        self.calls += 1
        raise AgentModelError("Intern connection failed")


def _quality_request(
    *,
    message: str = MESSAGE,
    status: str = "ok",
    dataset_id: str = "dataset-test",
    allow_tools: bool = True,
) -> MaterialAgentRequest:
    dataset = dict(_payload()["dataset"], dataset_id=dataset_id)
    inspection = {
        "dataset": dataset,
        "columns": [
            {
                "name": column,
                "inferred_type": "numeric",
                "non_null_count": 1000,
                "missing_count": 0,
                "unique_count": 7,
            }
            for column in [*COLUMNS, "crystal_system"]
        ],
        "preview_offset": 0,
        "preview_rows": [],
        "duplicate_row_count": 0,
    }
    envelope = {
        "status": status,
        "call_id": "quality",
        "tool_name": "assess_data_quality",
        "evidence_id": "evidence-quality",
        "output": {
            "evidence_id": "evidence-quality",
            "dataset": dataset,
            "inspection": inspection,
        },
    }
    if status == "error":
        envelope.update(
            output=None,
            error={
                "code": "TOOL_FAILED",
                "message": "dataset missing",
                "retryable": False,
            },
        )
    return MaterialAgentRequest(
        instructions="analysis",
        tool_definitions=(),
        final_draft_schema={},
        allow_tool_calls=allow_tools,
        input_items=(
            AgentMessageItem(role="user", content=message),
            AgentFunctionCallItem(
                call_id="quality",
                name="assess_data_quality",
                arguments=json.dumps({"dataset_id": dataset_id}),
            ),
            AgentFunctionOutputItem(call_id="quality", output=json.dumps(envelope)),
        ),
    )


def test_quality_success_routes_describe_without_remote_model() -> None:
    delegate = _OfflineModel()
    response = DataAnalysisRoutingModel(delegate).generate(_quality_request())
    call = response.tool_calls[0]
    assert call.name == "describe_dataset"
    assert json.loads(call.arguments) == {
        "dataset_id": "dataset-test",
        "columns": COLUMNS,
        "group_by": "crystal_system",
        "quantiles": [0.25, 0.5, 0.75],
    }
    assert call.call_id != "quality"
    assert delegate.calls == 0


def test_explicit_description_starts_with_quality_check_without_model() -> None:
    delegate = _OfflineModel()
    request = MaterialAgentRequest(
        instructions="analysis",
        tool_definitions=(),
        final_draft_schema={},
        input_items=(
            AgentMessageItem(
                role="user",
                content=(
                    "描述 dataset-test columns=band_gap_ev,density_g_cm3,"
                    "energy_above_hull_ev_atom group_by=crystal_system"
                ),
            ),
        ),
    )
    response = DataAnalysisRoutingModel(delegate).generate(request)
    assert response.tool_calls[0].name == "assess_data_quality"
    assert delegate.calls == 0


def test_failed_quality_does_not_launch_statistics_or_hide_failure() -> None:
    delegate = _OfflineModel()
    response = DataAnalysisRoutingModel(delegate).generate(
        _quality_request(status="error")
    )
    assert response.tool_calls == ()
    draft = json.loads(response.message_text)
    assert draft["status"] == "error"
    assert "dataset missing" in draft["answer"]
    assert delegate.calls == 0


@pytest.mark.parametrize(
    "message",
    [
        "对 dataset-test 做描述统计，报告缺失值。",  # No explicit fields.
        MESSAGE.replace("group_by=crystal_system", "按晶系分组"),
        MESSAGE.replace(
            "columns=band_gap_ev,density_g_cm3,energy_above_hull_ev_atom", "columns="
        ),
        MESSAGE.replace("group_by=crystal_system", "group_by="),
        MESSAGE.replace(
            "columns=band_gap_ev,density_g_cm3,energy_above_hull_ev_atom",
            "columns=band_gap_ev,,density_g_cm3",
        ),
        MESSAGE.replace(
            "columns=band_gap_ev,density_g_cm3,energy_above_hull_ev_atom",
            "columns=band_gap_ev,band_gap_ev",
        ),
        MESSAGE + "另做Pearson相关分析。",
        MESSAGE + "并生成散点图。",
        MESSAGE + "导出CSV。",
        MESSAGE + " quantiles=0.1,0.9",
        MESSAGE + "再分析 dataset-second。",
        MESSAGE.replace("dataset-test", "dataset-other"),
    ],
)
def test_incomplete_or_complex_request_keeps_model_selection(message: str) -> None:
    delegate = _BadSummaryModel()
    response = DataAnalysisRoutingModel(delegate).generate(
        _quality_request(message=message)
    )
    assert response.tool_calls == ()
    assert delegate.calls == 1


def test_unrelated_quality_output_does_not_trigger_statistics() -> None:
    delegate = _BadSummaryModel()
    DataAnalysisRoutingModel(delegate).generate(
        _quality_request(dataset_id="dataset-other")
    )
    assert delegate.calls == 1


def test_old_turn_quality_output_is_not_used_as_current_check() -> None:
    old = _quality_request()
    request = MaterialAgentRequest(
        instructions="analysis",
        tool_definitions=(),
        final_draft_schema={},
        input_items=(*old.input_items, AgentMessageItem(role="user", content=MESSAGE)),
    )
    delegate = _OfflineModel()
    response = DataAnalysisRoutingModel(delegate).generate(request)
    assert response.tool_calls[0].name == "assess_data_quality"
    assert delegate.calls == 0


def test_disabled_tool_calls_do_not_launch_another_operation() -> None:
    delegate = _OfflineModel()
    response = DataAnalysisRoutingModel(delegate).generate(
        _quality_request(allow_tools=False)
    )
    assert response.tool_calls == ()
    assert json.loads(response.message_text)["status"] == "error"
    assert delegate.calls == 0


def test_existing_description_attempt_is_not_automatically_repeated() -> None:
    quality = _quality_request()
    request = MaterialAgentRequest(
        instructions="analysis",
        tool_definitions=(),
        final_draft_schema={},
        input_items=(
            *quality.input_items,
            AgentFunctionCallItem(
                call_id="already-description",
                name="describe_dataset",
                arguments=json.dumps(
                    {
                        "dataset_id": "dataset-test",
                        "columns": COLUMNS,
                        "group_by": "crystal_system",
                    }
                ),
            ),
        ),
    )
    delegate = _BadSummaryModel()
    response = DataAnalysisRoutingModel(delegate).generate(request)
    assert response.tool_calls == ()
    assert delegate.calls == 1


@pytest.mark.parametrize("damage", ["evidence", "dataset", "inspection", "call_id"])
def test_invalid_quality_evidence_cannot_trigger_statistics(damage: str) -> None:
    original = _quality_request()
    envelope = json.loads(original.input_items[-1].output)
    if damage == "evidence":
        envelope["output"]["evidence_id"] = "evidence-other"
    elif damage == "dataset":
        envelope["output"]["dataset"]["dataset_id"] = "dataset-other"
    elif damage == "inspection":
        envelope["output"]["inspection"] = None
    else:
        envelope["call_id"] = "other-call"
    request = MaterialAgentRequest(
        instructions="analysis",
        tool_definitions=(),
        final_draft_schema={},
        input_items=(
            *original.input_items[:-1],
            AgentFunctionOutputItem(
                call_id="quality",
                output=json.dumps(envelope),
            ),
        ),
    )
    delegate = _OfflineModel()
    response = DataAnalysisRoutingModel(delegate).generate(request)
    assert response.tool_calls == ()
    assert json.loads(response.message_text)["status"] == "error"
    assert delegate.calls == 0
