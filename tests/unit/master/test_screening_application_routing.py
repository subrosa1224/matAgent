"""Research topics must survive the database-to-literature handoff."""

import json
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from materials_screening.agent.intern_model import _forced_literature_search_tool
from materials_screening.agent.model_base import MaterialAgentRequest
from materials_screening.agent.models import AgentMessageItem
from materials_screening.agent.tool_base import AgentToolDefinition, ToolSideEffect
from materials_screening.master.application_feasibility import ApplicationCandidate
from materials_screening.master.master_nodes import (
    _material_screening_chain_passthrough,
    call_master_model_node,
)


def _state(application: str, *, include_literature: bool = False) -> dict:
    state = {
        "user_message": (
            f"筛选用于{application}的氧化物，分析候选带隙分布，"
            "并结合实验文献评估前5名的应用可行性。"
        ),
        "executed_call_ids": ["db", "da"],
        "sub_agent_call_count": 2,
        "sub_agent_results": [
            {
                "call_id": "db",
                "sub_agent_name": "materials_database",
                "status": "ok",
                "response_text": (
                    "| mp-1 | Fe2O3 |\nMATERIAL_QUERY_HANDOFF: query_id=query-test"
                ),
                "warnings": [],
            },
            {
                "call_id": "da",
                "sub_agent_name": "data_analysis",
                "status": "ok",
                "response_text": "候选带隙描述统计。",
                "warnings": [],
            },
        ],
    }
    if include_literature:
        state["executed_call_ids"].append("lit")
        state["sub_agent_results"].append(
            {
                "call_id": "lit",
                "sub_agent_name": "literature",
                "status": "ok",
                "response_text": "找到 Fe2O3 相关论文题名与摘要，尚未核验全文。",
                "warnings": ["摘要不等于全文"],
            }
        )
    return state


def _coordinator() -> Mock:
    coordinator = Mock()
    coordinator.material_query_candidates.return_value = (
        {"material_id": "mp-1", "formula_pretty": "Fe2O3"},
        {"material_id": "mp-2", "formula_pretty": "Fe2O3"},
        {"material_id": "mp-3", "formula_pretty": "WO3"},
    )
    coordinator.prioritized_material_query_candidates.return_value = (
        ApplicationCandidate(
            priority=1,
            material_id="mp-4",
            formula_pretty="ZnO",
            elements=("Zn", "O"),
            band_gap_ev=3.0,
            energy_above_hull_ev_atom=0.0,
            density_g_cm3=5.0,
            theoretical=False,
            is_gap_direct=True,
            risk_level="standard",
        ),
    )
    return coordinator


def test_database_handoff_keeps_formula_and_crystal_independent_groups() -> None:
    state = _state("锂离子电池正极")
    state["user_message"] += "分析不同化学式和晶系的稳定性、计算带隙和密度分布。"
    state["sub_agent_results"] = state["sub_agent_results"][:1]
    state["executed_call_ids"] = ["db"]
    state["sub_agent_call_count"] = 1
    coordinator = _coordinator()
    coordinator.material_query_to_analysis.return_value = SimpleNamespace(
        partition=SimpleNamespace(dataset_id="dataset-test")
    )
    context = SimpleNamespace(
        cancel_event=None,
        clock=lambda: datetime.now(UTC),
        settings=SimpleNamespace(agent_max_tool_calls_per_turn=3),
        id_generator=SimpleNamespace(new_id=lambda: "route-test"),
        sub_agent_registry=SimpleNamespace(
            resolve_by_delegate_function=lambda name: SimpleNamespace(
                delegate_function_name=name
            )
        ),
        data_analysis_coordinator=coordinator,
    )
    result = call_master_model_node(state, SimpleNamespace(context=context))
    call = result["pending_tool_calls"][0]
    assert call["name"] == "delegate_to_data_analysis"
    task = json.loads(call["arguments_json"])["task"]
    assert "group_by_each=crystal_system,formula_pretty" in task
    assert "不推断器件性能" in task


@pytest.mark.parametrize(
    "application", ["可见光光催化", "紫外光催化", "锂离子电池正极"]
)
def test_general_handoff_keeps_question_and_avoids_uv_rules(application: str) -> None:
    state = _state(application)
    coordinator = _coordinator()
    context = SimpleNamespace(
        cancel_event=None,
        clock=lambda: datetime.now(UTC),
        settings=SimpleNamespace(agent_max_tool_calls_per_turn=5),
        id_generator=SimpleNamespace(new_id=lambda: "route-test"),
        sub_agent_registry=SimpleNamespace(
            resolve_by_delegate_function=lambda name: SimpleNamespace(
                delegate_function_name=name
            )
        ),
        data_analysis_coordinator=coordinator,
    )
    result = call_master_model_node(state, SimpleNamespace(context=context))
    call = result["pending_tool_calls"][0]
    task = json.loads(call["arguments_json"])["task"]
    assert call["name"] == "delegate_to_literature"
    assert state["user_message"] in task
    assert "Fe2O3" in task and "WO3" in task
    coordinator.material_query_candidates.assert_called_once_with(
        "query-test", limit=20, unique_formulas=True
    )
    assert "先按化学式去重" in task
    assert "凸包能" in task and "不是应用性能排名" in task
    assert "候选池文献预检" not in task  # This phrase forces the UV-specific tool.
    assert "紫外光电探测" not in task
    assert "A/B/C/NONE" not in task
    coordinator.prioritized_material_query_candidates.assert_not_called()
    request = MaterialAgentRequest(
        instructions="literature",
        input_items=(AgentMessageItem(role="user", content=task),),
        tool_definitions=tuple(
            AgentToolDefinition(
                name=name,
                description="test tool",
                parameters={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                side_effect=ToolSideEffect.READ_ONLY,
                version="1",
            )
            for name in ("literature_search", "screen_candidate_literature")
        ),
        final_draft_schema={"type": "object"},
    )
    assert _forced_literature_search_tool(request) == "literature_search"


def test_uv_detector_handoff_keeps_specialized_screening() -> None:
    state = _state("紫外光电探测")
    coordinator = _coordinator()
    context = SimpleNamespace(
        cancel_event=None,
        clock=lambda: datetime.now(UTC),
        settings=SimpleNamespace(agent_max_tool_calls_per_turn=5),
        id_generator=SimpleNamespace(new_id=lambda: "route-test"),
        sub_agent_registry=SimpleNamespace(
            resolve_by_delegate_function=lambda name: SimpleNamespace(
                delegate_function_name=name
            )
        ),
        data_analysis_coordinator=coordinator,
    )
    result = call_master_model_node(state, SimpleNamespace(context=context))
    task = json.loads(result["pending_tool_calls"][0]["arguments_json"])["task"]
    assert "候选池文献预检" in task
    assert "A/B/C/NONE" in task
    coordinator.material_query_candidates.assert_not_called()


def test_general_report_does_not_invent_uv_ranking_or_completion() -> None:
    coordinator = _coordinator()
    draft = _material_screening_chain_passthrough(
        _state("可见光光催化", include_literature=True), coordinator=coordinator
    )
    assert draft is not None
    assert "紫外" not in draft["answer"]
    assert "可见光光催化" in draft["answer"]
    assert "不是最终推荐" in draft["answer"]
    assert "摘要不等于全文" in draft["warnings"]
    assert draft["status"] == "needs_user_input"
    assert "PDF" in draft["follow_up_question"]
    assert draft["evidence_ids"] == ["db", "da", "lit"]
    coordinator.prioritized_material_query_candidates.assert_not_called()


def test_general_literature_failure_is_visible_without_uv_conclusion() -> None:
    state = _state("可见光光催化", include_literature=True)
    state["sub_agent_results"][-1].update(
        status="error", response_text="", error={"message": "providers failed"}
    )
    draft = _material_screening_chain_passthrough(state)
    assert draft is not None
    assert "紫外" not in draft["answer"]
    assert "执行失败" in draft["answer"]
    assert "providers failed" in draft["warnings"]
    assert draft["status"] == "error"
