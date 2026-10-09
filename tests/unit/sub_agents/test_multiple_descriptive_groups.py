import copy
import json
from dataclasses import replace

import pytest
from tests.unit.sub_agents.test_descriptive_report_routing import (
    MESSAGE,
    _payload,
    _request,
)
from tests.unit.sub_agents.test_descriptive_tool_handoff import (
    _OfflineModel,
    _quality_request,
)

from materials_screening.agent.model_base import MaterialAgentRequest
from materials_screening.agent.models import (
    AgentFunctionOutputItem,
)
from materials_screening.sub_agents.data_analysis.routing_model import (
    DataAnalysisRoutingModel,
)

MULTIPLE = MESSAGE.replace(
    "group_by=crystal_system", "group_by_each=crystal_system,formula_pretty"
)


def test_independent_groups_wait_for_both_statistics_without_remote_model() -> None:
    delegate = _OfflineModel()
    model = DataAnalysisRoutingModel(delegate)
    quality = _quality_request(message=MULTIPLE)
    first = model.generate(quality).tool_calls[0]
    assert json.loads(first.arguments)["group_by"] == "crystal_system"
    first_output = _request(_payload(), message=MULTIPLE).input_items[-1]
    envelope = json.loads(first_output.output)
    envelope["call_id"] = first.call_id
    request = MaterialAgentRequest(
        instructions="analysis",
        tool_definitions=(),
        final_draft_schema={},
        input_items=(
            *quality.input_items,
            first,
            AgentFunctionOutputItem(call_id=first.call_id, output=json.dumps(envelope)),
        ),
    )
    second_response = model.generate(request)
    assert not second_response.message_text
    second = second_response.tool_calls[0]
    assert json.loads(second.arguments)["group_by"] == "formula_pretty"
    assert first.call_id != second.call_id
    payload = copy.deepcopy(_payload())
    payload["evidence_id"] = "evidence-formula"
    payload["analysis"]["evidence_id"] = "evidence-formula"
    payload["analysis"]["analysis_id"] = "analysis-formula"
    payload["analysis"]["parameters"]["group_by"] = "formula_pretty"
    envelope.update(
        call_id=second.call_id, evidence_id="evidence-formula", output=payload
    )
    final_request = replace(
        request,
        input_items=(
            *request.input_items,
            second,
            AgentFunctionOutputItem(
                call_id=second.call_id, output=json.dumps(envelope)
            ),
        ),
    )
    draft = json.loads(model.generate(final_request).message_text)
    assert draft["status"] == "completed"
    assert "crystal_system" in draft["answer"] and "formula_pretty" in draft["answer"]
    assert draft["evidence_ids"] == ["evidence-description", "evidence-formula"]
    assert delegate.calls == 0


def test_failed_first_dimension_does_not_claim_both_complete() -> None:
    delegate = _OfflineModel()
    request = _request(_payload(), message=MULTIPLE, status="error")
    draft = json.loads(
        DataAnalysisRoutingModel(delegate).generate(request).message_text
    )
    assert draft["status"] == "error"
    assert "unknown column" in draft["answer"]
    assert delegate.calls == 0


def test_missing_return_is_not_retried_or_rendered_as_complete() -> None:
    model = DataAnalysisRoutingModel(_OfflineModel())
    quality = _quality_request(message=MULTIPLE)
    attempted = model.generate(quality).tool_calls[0]
    request = replace(quality, input_items=(*quality.input_items, attempted))
    response = model.generate(request)
    assert not response.tool_calls
    assert json.loads(response.message_text)["status"] == "error"


@pytest.mark.parametrize(
    "grouping",
    [
        "group_by_each=",
        "group_by_each=crystal_system,crystal_system",
        "group_by_each=crystal_system,,formula_pretty",
        "group_by_each=crystal_system,formula_pretty group_by=crystal_system",
    ],
)
def test_malformed_independent_grouping_cannot_silently_become_overall(
    grouping: str,
) -> None:
    from tests.unit.sub_agents.test_descriptive_report_routing import _BadSummaryModel

    delegate = _BadSummaryModel()
    message = MESSAGE.replace("group_by=crystal_system", grouping)
    response = DataAnalysisRoutingModel(delegate).generate(
        _quality_request(message=message)
    )
    assert not response.tool_calls
    assert delegate.calls == 1
