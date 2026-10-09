"""Tests for deterministic outlier sub-agent tool routing."""

from __future__ import annotations

import json
from unittest.mock import Mock

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
from materials_screening.sub_agents.outlier_detection.routing_model import (
    OutlierDetectionRoutingModel,
)


def _request(*items: object) -> MaterialAgentRequest:
    return MaterialAgentRequest(
        instructions="outlier agent",
        input_items=items,  # type: ignore[arg-type]
        tool_definitions=(),
        final_draft_schema={"type": "object"},
    )


def test_first_turn_routes_to_the_only_tool_without_calling_llm() -> None:
    final_model = Mock()
    model = OutlierDetectionRoutingModel(final_model)
    query = "比较 TiO2 和 SiO2 的带隙，找出离群材料"

    response = model.generate(_request(AgentMessageItem(role="user", content=query)))

    call = response.output_items[0]
    assert isinstance(call, AgentFunctionCallItem)
    assert call.name == "run_outlier_detection"
    assert json.loads(call.arguments) == {"query": query}
    final_model.generate.assert_not_called()


def test_after_tool_result_delegates_final_answer_to_llm() -> None:
    expected = MaterialAgentResponse(
        status=AgentModelStatus.COMPLETED,
        output_items=(AgentMessageItem(role="assistant", content="{}"),),
        request_id="intern-1",
        provider="intern",
        model="intern-s2-preview-35b",
    )
    final_model = Mock()
    final_model.generate.return_value = expected
    model = OutlierDetectionRoutingModel(final_model)
    request = _request(
        AgentMessageItem(role="user", content="compare TiO2 and SiO2"),
        AgentFunctionCallItem(
            call_id="call-1", name="run_outlier_detection", arguments="{}"
        ),
        AgentFunctionOutputItem(
            call_id="call-1", output='{"status":"ok","evidence_id":"evt-1"}'
        ),
    )

    assert model.generate(request) is expected
    final_model.generate.assert_called_once_with(request)
