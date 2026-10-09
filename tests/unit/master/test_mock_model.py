"""Unit tests for the deterministic Master mock model."""

from __future__ import annotations

import json

from materials_screening.agent.model_base import MaterialAgentRequest
from materials_screening.agent.models import (
    AgentFunctionCallItem,
    AgentFunctionOutputItem,
    AgentMessageItem,
)
from materials_screening.master.mock_model import MasterMockAgentModel


def _request(*items: object) -> MaterialAgentRequest:
    return MaterialAgentRequest(
        input_items=items,  # type: ignore[arg-type]
        instructions="master",
        tool_definitions=(),
        final_draft_schema={"type": "object"},
    )


def test_outlier_request_delegates_to_database_agent() -> None:
    response = MasterMockAgentModel().generate(
        _request(AgentMessageItem(role="user", content="检测离群材料"))
    )
    call = response.tool_calls[0]
    assert call.name == "delegate_to_materials_database"
    assert json.loads(call.arguments) == {"task": "检测离群材料"}


def test_literature_request_delegates_to_literature_agent() -> None:
    response = MasterMockAgentModel().generate(
        _request(
            AgentMessageItem(role="user", content="检索二氧化钛光催化相关文献和 DOI")
        )
    )
    call = response.tool_calls[0]
    assert call.name == "delegate_to_literature"


def test_dataset_request_delegates_to_data_analysis_before_database() -> None:
    response = MasterMockAgentModel().generate(
        _request(
            AgentMessageItem(
                role="user",
                content="统计 dataset-12345678 的缺失值和相关性",
            )
        )
    )
    assert response.tool_calls[0].name == "delegate_to_data_analysis"


def test_sub_agent_result_becomes_master_final_answer() -> None:
    response = MasterMockAgentModel().generate(
        _request(
            AgentMessageItem(role="user", content="检测离群材料"),
            AgentFunctionCallItem(
                call_id="call-1",
                name="delegate_to_outlier_detection",
                arguments='{"task":"检测离群材料"}',
            ),
            AgentFunctionOutputItem(
                call_id="call-1",
                output=json.dumps(
                    {
                        "status": "ok",
                        "response_text": "检测完成。",
                        "evidence_ids": ["evt-1"],
                        "warnings": [],
                    }
                ),
            ),
        )
    )
    draft = json.loads(response.message_text)
    assert draft["status"] == "completed"
    assert draft["answer"] == "检测完成。"
    assert draft["evidence_ids"] == ["evt-1"]


def test_ambiguous_request_asks_for_safe_clarification() -> None:
    response = MasterMockAgentModel().generate(
        _request(AgentMessageItem(role="user", content="帮我看看这个"))
    )
    assert not response.tool_calls
    draft = json.loads(response.message_text)
    assert draft["status"] == "needs_user_input"
    assert "材料数据库" in draft["follow_up_question"]
