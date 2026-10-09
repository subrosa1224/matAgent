"""Regression tests for complete standard oxide filters."""

from __future__ import annotations

import json

from materials_screening.agent.model_base import (
    AgentModelStatus,
    MaterialAgentRequest,
    MaterialAgentResponse,
)
from materials_screening.agent.models import AgentFunctionOutputItem, AgentMessageItem
from materials_screening.sub_agents.materials_database.deterministic_model import (
    DeterministicDatabaseModel,
    _band_gap_exploration_search,
    parse_standard_search,
)


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


def _request(*items: object) -> MaterialAgentRequest:
    return MaterialAgentRequest(
        input_items=items,  # type: ignore[arg-type]
        instructions="database",
        tool_definitions=(),
        final_draft_schema={"type": "object"},
    )


def test_complete_oxide_filter_becomes_structured_search() -> None:
    search = parse_standard_search("筛选带隙大于2 eV的稳定氧化物")
    assert search is not None
    assert search.required_elements == ("O",)
    assert search.formula is None
    assert [
        (item.field, item.operator.value, item.value) for item in search.filters
    ] == [
        ("band_gap_ev", "gt", 2.0),
        ("is_stable", "eq", True),
    ]
    assert {"F", "Cl", "Br", "N", "S"}.issubset(search.excluded_elements)


def test_complete_filter_calls_search_without_model_clarification() -> None:
    response = DeterministicDatabaseModel(_Delegate()).generate(
        _request(AgentMessageItem(role="user", content="筛选带隙大于2 eV的稳定氧化物"))
    )
    assert response.tool_calls[0].name == "search_materials"
    arguments = json.loads(response.tool_calls[0].arguments)
    assert arguments["required_elements"] == ["O"]
    assert {"F", "Cl", "Br", "N", "S"}.issubset(arguments["excluded_elements"])


def test_bounded_uv_oxide_screening_becomes_structured_search() -> None:
    search = parse_standard_search(
        "筛选带隙为2.8～4.5 eV、凸包能不高于0.05 eV/atom且"
        "不含Pb、Cd、Hg的氧化物半导体，分析稳定性和密度分布。"
    )

    assert search is not None
    assert search.required_elements == ("O",)
    assert {"Pb", "Cd", "Hg", "F", "P", "N", "S"}.issubset(search.excluded_elements)
    assert [
        (item.field, item.operator.value, item.value) for item in search.filters
    ] == [
        ("band_gap_ev", "gte", 2.8),
        ("band_gap_ev", "lte", 4.5),
        ("energy_above_hull_ev_atom", "lte", 0.05),
        ("is_metal", "eq", False),
    ]
    assert search.sort[0].field == "energy_above_hull_ev_atom"
    assert "density_g_cm3" in search.fields


def test_tool_result_becomes_direct_evidence_grounded_table() -> None:
    output = {
        "status": "ok",
        "evidence_id": "evidence-1",
        "output": {
            "query_id": "query-1",
            "matched_count": 1,
            "materials": [
                {
                    "material_id": "mp-1",
                    "formula_pretty": "TiO2",
                    "band_gap_ev": 3.1,
                    "is_stable": True,
                    "energy_above_hull_ev_atom": 0.0,
                }
            ],
            "warnings": [],
        },
    }
    response = DeterministicDatabaseModel(_Delegate()).generate(
        _request(
            AgentMessageItem(role="user", content="筛选带隙大于2 eV的稳定氧化物"),
            AgentFunctionOutputItem(call_id="call-1", output=json.dumps(output)),
        )
    )
    draft = json.loads(response.message_text)
    assert draft["status"] == "completed"
    assert "mp-1" in draft["answer"]
    assert "请确认" not in draft["answer"]
    assert draft["evidence_ids"] == ["evidence-1"]
    assert "MATERIAL_QUERY_HANDOFF: query_id=query-1" in draft["answer"]


def test_incomplete_or_different_request_still_uses_delegate() -> None:
    response = DeterministicDatabaseModel(_Delegate()).generate(
        _request(AgentMessageItem(role="user", content="帮我找一些氧化物"))
    )
    assert response.message_text == "delegate"


def test_exploration_preserves_strict_query_and_other_constraints() -> None:
    strict = parse_standard_search(
        "筛选带隙为2.8～4.5 eV、凸包能不高于0.05 eV/atom且不含Pb、Cd、Hg的氧化物"
    )
    assert strict is not None
    exploration = _band_gap_exploration_search(strict)
    assert exploration is not None
    assert strict.num_elements is None
    assert exploration.num_elements == 2
    assert exploration.excluded_elements == strict.excluded_elements
    assert any(
        item.field == "band_gap_ev"
        and item.operator.value == "gte"
        and item.value == 2.8
        for item in strict.filters
    )
    assert not any(
        item.field == "band_gap_ev" and item.operator.value == "gte"
        for item in exploration.filters
    )
    assert any(
        item.field == "energy_above_hull_ev_atom" and item.value == 0.05
        for item in exploration.filters
    )


def test_exploration_calls_second_query_and_keeps_distinct_snapshot() -> None:
    message = (
        "筛选带隙为2.8～4.5 eV、凸包能不高于0.05 eV/atom的氧化物。"
        "BAND_GAP_EXPLORATION_POOL"
    )
    first = AgentFunctionOutputItem(
        call_id="first",
        output=json.dumps(
            {
                "status": "ok",
                "evidence_id": "strict-evidence",
                "output": {"query_id": "query-strict", "materials": []},
            }
        ),
    )
    model = DeterministicDatabaseModel(_Delegate())
    user = AgentMessageItem(role="user", content=message)
    response = model.generate(_request(user, first))
    assert response.tool_calls[0].name == "search_materials"
    assert json.loads(response.tool_calls[0].arguments)["num_elements"] == 2
    second = AgentFunctionOutputItem(
        call_id="second",
        output=json.dumps(
            {
                "status": "ok",
                "evidence_id": "explore-evidence",
                "output": {"query_id": "query-exploration", "materials": []},
            }
        ),
    )
    response = model.generate(_request(user, first, second))
    draft = json.loads(response.message_text)
    assert "MATERIAL_QUERY_HANDOFF: query_id=query-strict" in draft["answer"]
    assert "SUPPLEMENTARY_QUERY_HANDOFF: query_id=query-exploration" in draft["answer"]
    assert draft["evidence_ids"] == ["strict-evidence", "explore-evidence"]
