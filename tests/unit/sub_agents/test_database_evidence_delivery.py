"""Final database answers must retain tool evidence, not model extrapolations."""

import json

import pytest

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
from materials_screening.sub_agents.materials_database.deterministic_model import (
    DeterministicDatabaseModel,
)


class Delegate:
    def __init__(self, *, tool_call: bool = False, final_status: str = "completed"):
        self.tool_call = tool_call
        self.final_status = final_status

    def generate(self, request):
        if self.tool_call:
            return MaterialAgentResponse(
                status=AgentModelStatus.COMPLETED,
                output_items=(
                    AgentFunctionCallItem(
                        call_id="next",
                        name="describe_materials",
                        arguments="{}",
                    ),
                ),
                request_id="test",
                provider="test",
                model="test",
            )
        draft = {
            "status": self.final_status,
            "answer": "带隙太小，电压平台不足；近稳定材料4个。",
            "warnings": ["此材料不适合做正极。"],
            "referenced_material_ids": [],
            "evidence_ids": [],
            "follow_up_question": None,
        }
        return MaterialAgentResponse(
            status=AgentModelStatus.COMPLETED,
            output_items=(
                AgentMessageItem(
                    role="assistant",
                    content=json.dumps(draft, ensure_ascii=False),
                ),
            ),
            request_id="test",
            provider="test",
            model="test",
        )


def request(*items):
    return MaterialAgentRequest(
        instructions="database",
        input_items=items,
        tool_definitions=(),
        final_draft_schema={"type": "object"},
    )


def result(call_id, name, payload, *, status="ok"):
    return (
        AgentFunctionCallItem(call_id=call_id, name=name, arguments="{}"),
        AgentFunctionOutputItem(
            call_id=call_id,
            output=json.dumps(
                {
                    "status": status,
                    "tool_name": name,
                    "evidence_id": f"e-{call_id}",
                    "output": payload,
                }
            ),
        ),
    )


def search_payload(query_id="query-1"):
    return {
        "query_id": query_id,
        "matched_count": 84,
        "returned_count": 1,
        "fields": ["material_id", "formula_pretty", "band_gap_ev"],
        "materials": [
            {"material_id": "mp-1", "formula_pretty": "LiFeO2", "band_gap_ev": 1.62}
        ],
        "warnings": ["Only the returned page is displayed."],
    }


USER = AgentMessageItem(role="user", content="筛选Li-Fe-O，分析并评估正极文献。")


def test_generic_query_keeps_handoff_and_drops_unsupported_model_conclusions():
    response = DeterministicDatabaseModel(Delegate()).generate(
        request(
            USER,
            *result("s1", "search_materials", search_payload()),
        )
    )
    draft = json.loads(response.message_text)
    assert "MATERIAL_QUERY_HANDOFF: query_id=query-1" in draft["answer"]
    assert "mp-1" in draft["answer"] and "LiFeO2" in draft["answer"]
    assert "84" in draft["answer"]
    assert "电压平台不足" not in draft["answer"]
    assert "近稳定材料4个" not in draft["answer"]
    assert "不适合做正极" not in " ".join(draft["warnings"])
    assert "Only the returned page" in " ".join(draft["warnings"])
    assert draft["evidence_ids"] == ["e-s1"]
    assert draft["referenced_material_ids"] == ["mp-1"]


def test_statistics_are_rendered_from_the_actual_tool_result():
    response = DeterministicDatabaseModel(Delegate()).generate(
        request(
            USER,
            *result("s1", "search_materials", search_payload()),
            *result(
                "d1",
                "describe_materials",
                {
                    "query_id": "query-1",
                    "analysis_id": "analysis-1",
                    "statistics": {
                        "band_gap_ev": {
                            "count": 84,
                            "missing": 0,
                            "mean": 1.119,
                            "median": 0.895,
                        }
                    },
                },
            ),
        )
    )
    draft = json.loads(response.message_text)
    assert "analysis-1" in draft["answer"] and "1.119" in draft["answer"]
    assert draft["evidence_ids"] == ["e-s1", "e-d1"]


def test_later_tool_call_is_not_replaced_with_a_premature_final():
    response = DeterministicDatabaseModel(Delegate(tool_call=True)).generate(
        request(
            USER,
            *result("s1", "search_materials", search_payload()),
        )
    )
    assert response.tool_calls[0].name == "describe_materials"


def test_old_turn_and_failed_search_cannot_provide_a_handoff():
    response = DeterministicDatabaseModel(Delegate()).generate(
        request(
            *result("old", "search_materials", search_payload("query-old")),
            USER,
            *result("failed", "search_materials", None, status="error"),
        )
    )
    assert "MATERIAL_QUERY_HANDOFF" not in response.message_text


def test_distinct_successful_queries_are_not_silently_merged():
    response = DeterministicDatabaseModel(Delegate()).generate(
        request(
            USER,
            *result("s1", "search_materials", search_payload("query-1")),
            *result("s2", "search_materials", search_payload("query-2")),
        )
    )
    draft = json.loads(response.message_text)
    assert draft["status"] != "completed"
    assert "MATERIAL_QUERY_HANDOFF" not in draft["answer"]
    assert "电压平台不足" not in draft["answer"]


@pytest.mark.parametrize("status", ["error", "needs_user_input"])
def test_terminal_error_or_clarification_is_not_promoted_to_success(status):
    response = DeterministicDatabaseModel(Delegate(final_status=status)).generate(
        request(
            USER,
            *result("s1", "search_materials", search_payload()),
        )
    )
    assert json.loads(response.message_text)["status"] == status


def test_full_tool_counts_are_not_recomputed_from_the_display_page():
    response = DeterministicDatabaseModel(Delegate()).generate(
        request(
            USER,
            *result("s1", "search_materials", search_payload()),
            *result(
                "p1",
                "get_query_result",
                {
                    **search_payload(),
                    "matched_count": 84,
                    "materials": [
                        {
                            "material_id": "mp-2",
                            "formula_pretty": "Li2FeO3",
                            "band_gap_ev": 0.1,
                        }
                    ],
                },
            ),
        )
    )
    draft = json.loads(response.message_text)
    assert "84" in draft["answer"]
    assert "mean" not in draft["answer"]
    assert "1.119" not in draft["answer"]


def test_real_pagination_shape_preserves_search_count_and_returned_properties():
    response = DeterministicDatabaseModel(Delegate()).generate(
        request(
            USER,
            *result("s1", "search_materials", search_payload()),
            *result(
                "p1",
                "get_query_result",
                {
                    "query_id": "query-1",
                    "offset": 0,
                    "limit": 100,
                    "total": 84,
                    "materials": search_payload()["materials"],
                },
            ),
        )
    )
    draft = json.loads(response.message_text)
    assert "匹配 84" in draft["answer"]
    assert "band_gap_ev" in draft["answer"] and "1.62" in draft["answer"]


def test_malformed_query_identifier_is_not_used_for_handoff():
    payload = {**search_payload(), "query_id": {"unexpected": "object"}}
    response = DeterministicDatabaseModel(Delegate()).generate(
        request(USER, *result("s1", "search_materials", payload))
    )
    assert "MATERIAL_QUERY_HANDOFF" not in response.message_text
