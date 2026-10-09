"""Repeated searches must read evidence this turn, never publish stale narrative."""

import json
from dataclasses import replace

import pytest
from tests.unit.sub_agents.test_database_evidence_delivery import (
    USER,
    Delegate,
    request,
    result,
    search_payload,
)

from materials_screening.agent.models import AgentMessageItem
from materials_screening.agent.tool_base import AgentToolDefinition, ToolSideEffect
from materials_screening.sub_agents.materials_database.deterministic_model import (
    DeterministicDatabaseModel,
)
from materials_screening.sub_agents.materials_database.models import GetQueryResultInput


def query_request(*items, allow=True):
    return replace(
        request(*items),
        allow_tool_calls=allow,
        tool_definitions=(
            AgentToolDefinition(
                name="get_query_result",
                description="Read saved query",
                parameters=GetQueryResultInput.model_json_schema(),
                side_effect=ToolSideEffect.READ_ONLY,
                version="1",
            ),
        ),
    )


def test_same_question_reads_matching_snapshot_before_any_model_final():
    response = DeterministicDatabaseModel(Delegate()).generate(
        query_request(
            USER,
            *result("old", "search_materials", search_payload("query-old")),
            AgentMessageItem(role="assistant", content="old final"),
            USER,
        )
    )
    assert response.tool_calls[0].name == "get_query_result"
    args = GetQueryResultInput.model_validate_json(response.tool_calls[0].arguments)
    assert args.query_id == "query-old"
    assert "band_gap_ev" in args.fields
    assert response.message_text is None


def test_repeat_of_a_repeat_preserves_fields_from_actual_page_keys():
    response = DeterministicDatabaseModel(Delegate()).generate(
        query_request(
            USER,
            *result(
                "old-read",
                "get_query_result",
                {
                    "query_id": "query-old",
                    "total": 84,
                    "materials": search_payload()["materials"],
                },
            ),
            USER,
        )
    )
    assert "band_gap_ev" in json.loads(response.tool_calls[0].arguments)["fields"]


def test_repeated_question_after_a_failed_repeat_can_read_earlier_valid_snapshot():
    response = DeterministicDatabaseModel(Delegate()).generate(
        query_request(
            USER,
            *result("old", "search_materials", search_payload("query-old")),
            USER,
            AgentMessageItem(role="assistant", content="ungrounded repeat"),
            USER,
        )
    )
    assert json.loads(response.tool_calls[0].arguments)["query_id"] == "query-old"


@pytest.mark.parametrize(
    "history",
    [
        (),
        (
            AgentMessageItem(role="user", content="筛选其他材料"),
            *result("old", "search_materials", search_payload("query-old")),
        ),
        (USER, *result("failed", "search_materials", None, status="error")),
        (
            USER,
            *result("old", "search_materials", search_payload("query-a")),
            *result("old2", "search_materials", search_payload("query-b")),
        ),
    ],
)
def test_no_unambiguous_matching_snapshot_means_explicit_error_not_invented_success(
    history,
):
    response = DeterministicDatabaseModel(Delegate()).generate(
        query_request(*history, USER)
    )
    assert not response.tool_calls
    draft = json.loads(response.message_text)
    assert draft["status"] == "error"
    assert "本轮" in draft["answer"]
    assert "电压平台不足" not in draft["answer"]
    assert not draft["evidence_ids"] and not draft["referenced_material_ids"]
    assert "MATERIAL_QUERY_HANDOFF" not in draft["answer"]


def test_failed_current_read_does_not_retry_or_publish_old_results():
    response = DeterministicDatabaseModel(Delegate()).generate(
        query_request(
            USER,
            *result("old", "search_materials", search_payload("query-old")),
            USER,
            *result("read", "get_query_result", None, status="error"),
        )
    )
    assert not response.tool_calls
    assert json.loads(response.message_text)["status"] == "error"


def test_successful_current_read_delivers_fresh_evidence_and_snapshot_total():
    payload = {
        "query_id": "query-old",
        "total": 84,
        "materials": search_payload()["materials"],
        "offset": 0,
        "limit": 100,
    }
    response = DeterministicDatabaseModel(Delegate()).generate(
        query_request(
            USER,
            *result("old", "search_materials", search_payload("query-old")),
            USER,
            *result("read", "get_query_result", payload),
        )
    )
    draft = json.loads(response.message_text)
    assert draft["status"] == "completed"
    assert draft["evidence_ids"] == ["e-read"]
    assert "快照保存 84" in draft["answer"]
    assert "不代表重新获取" in draft["answer"]
    assert "MATERIAL_QUERY_HANDOFF: query_id=query-old" in draft["answer"]


def test_tools_disabled_cannot_trigger_snapshot_read():
    response = DeterministicDatabaseModel(Delegate()).generate(
        query_request(
            USER,
            *result("old", "search_materials", search_payload("query-old")),
            USER,
            allow=False,
        )
    )
    assert not response.tool_calls
    assert json.loads(response.message_text)["status"] == "error"


def test_owned_repeat_read_finishes_from_tool_without_optional_model_clarification():
    class ForbiddenDelegate:
        def generate(self, request):
            pytest.fail(
                "A completed deterministic read must not need another model final"
            )

    response = DeterministicDatabaseModel(ForbiddenDelegate()).generate(
        query_request(
            USER,
            *result(
                "repeat-query-read",
                "get_query_result",
                {
                    "query_id": "query-old",
                    "total": 84,
                    "materials": search_payload()["materials"],
                },
            ),
        )
    )
    draft = json.loads(response.message_text)
    assert draft["status"] == "completed"
    assert draft["evidence_ids"] == ["e-repeat-query-read"]
    assert draft["follow_up_question"] is None


def test_owned_failed_read_is_error_without_another_model_call():
    class ForbiddenDelegate:
        def generate(self, request):
            pytest.fail("Failed deterministic reads must not be retried or invented")

    response = DeterministicDatabaseModel(ForbiddenDelegate()).generate(
        query_request(
            USER,
            *result("repeat-query-read", "get_query_result", None, status="error"),
        )
    )
    assert json.loads(response.message_text)["status"] == "error"


def test_nonsearch_explanation_keeps_normal_conversation_behavior():
    response = DeterministicDatabaseModel(Delegate()).generate(
        query_request(
            USER,
            *result("old", "search_materials", search_payload("query-old")),
            AgentMessageItem(role="user", content="解释一下凸包能的含义"),
        )
    )
    assert not response.tool_calls
    assert json.loads(response.message_text)["status"] == "completed"
