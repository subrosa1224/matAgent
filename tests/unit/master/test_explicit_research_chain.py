"""Explicit multi-stage requests must not depend on an evaluation keyword."""

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from materials_screening.master.master_nodes import (
    _requires_material_screening_chain,
    call_master_model_node,
)
from materials_screening.sub_agents.materials_database.screening_handoff import (
    SCREENING_HANDOFF_SCOPE,
)

SUITE = json.loads(
    (
        Path(__file__).resolve().parents[2]
        / "fixtures/research/stable_acceptance_v1/suite.json"
    ).read_text(encoding="utf-8")
)
QUESTIONS = [
    next(c["question"] for c in SUITE["cases"] if c["id"] == case_id)
    for case_id in ("N02", "N04", "N08")
]


@pytest.mark.parametrize("question", QUESTIONS, ids=("N02", "N04", "N08"))
def test_original_explicit_requests_are_recognized(question):
    assert _requires_material_screening_chain(question)


@pytest.mark.parametrize(
    "question",
    [
        "查询 ZnO 的计算性质，分析晶系分布，并检索实验论文。",
        "请查找 Fe2O3 候选，统计稳定性，结合实验文献形成报告。",
        "筛选氧化物，分析带隙，再做文献检索。",
    ],
)
def test_independent_requested_stages_do_not_need_evaluation_word(question):
    assert _requires_material_screening_chain(question)


@pytest.mark.parametrize(
    "question",
    [
        "查询 ZnO 的带隙。",
        "查询 ZnO 并统计带隙，不需要检索文献。",
        "查询 ZnO，不做分析，也不要检索论文。",
        "分析已上传的实验论文。",
        "请查询 ZnO 并统计密度。我的研究背景涉及文献。",
    ],
)
def test_single_stage_background_and_negated_tasks_do_not_force_chain(question):
    assert not _requires_material_screening_chain(question)


@pytest.mark.parametrize("question", QUESTIONS, ids=("N02", "N04", "N08"))
@pytest.mark.parametrize("available", [True, False], ids=("handoff", "unavailable"))
def test_database_success_alone_is_not_whole_task_completion(question, available):
    coordinator = Mock()
    coordinator.material_query_to_analysis.return_value = SimpleNamespace(
        partition=SimpleNamespace(dataset_id="dataset-explicit")
    )
    context = SimpleNamespace(
        cancel_event=None,
        clock=lambda: datetime.now(UTC),
        settings=SimpleNamespace(agent_max_tool_calls_per_turn=3),
        id_generator=SimpleNamespace(new_id=lambda: "analysis-next"),
        sub_agent_registry=SimpleNamespace(
            resolve_by_delegate_function=lambda name: SimpleNamespace(
                delegate_function_name=name
            )
        ),
        data_analysis_coordinator=coordinator if available else None,
    )
    state = {
        "user_message": question,
        "executed_call_ids": ["db"],
        "model_call_count": 1,
        "sub_agent_call_count": 1,
        "sub_agent_results": [
            {
                "call_id": "db",
                "sub_agent_name": "materials_database",
                "status": "ok",
                "response_text": "MATERIAL_QUERY_HANDOFF: query_id=query-explicit",
                "warnings": [],
            }
        ],
    }
    update = call_master_model_node(state, SimpleNamespace(context=context))
    if available:
        assert update["final_draft"] is None
        assert update["pending_tool_calls"][0]["name"] == "delegate_to_data_analysis"
    else:
        assert update["pending_tool_calls"] == []
        assert update["final_draft"]["status"] == "error"
        assert "未执行" in update["final_draft"]["answer"]


@pytest.mark.parametrize("question", QUESTIONS, ids=("N02", "N04", "N08"))
def test_original_first_delegation_keeps_scope_without_extra_candidate_pool(question):
    context = SimpleNamespace(
        cancel_event=None,
        clock=lambda: datetime.now(UTC),
        settings=SimpleNamespace(
            agent_max_tool_calls_per_turn=3,
            agent_max_model_calls_per_turn=6,
            agent_max_input_bytes=200000,
        ),
        id_generator=SimpleNamespace(new_id=lambda: "first-call"),
        sub_agent_registry=SimpleNamespace(
            definitions_for_model=lambda: [],
            resolve_by_delegate_function=lambda name: SimpleNamespace(
                delegate_function_name=name
            ),
        ),
        data_analysis_coordinator=None,
    )
    state = {"user_message": question, "model_call_count": 0, "input_items": []}
    update = call_master_model_node(state, SimpleNamespace(context=context))
    call = update["pending_tool_calls"][0]
    assert call["name"] == "delegate_to_materials_database"
    task = json.loads(call["arguments_json"])["task"]
    assert task == question + SCREENING_HANDOFF_SCOPE
    assert "BAND_GAP_EXPLORATION_POOL" not in task
