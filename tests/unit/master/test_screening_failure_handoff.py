"""Failures must not consume the slot reserved for independent literature work."""

import json
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from materials_screening.agent.models import AgentFinalDraft
from materials_screening.master.master_nodes import (
    call_master_model_node,
    finalize_success_node,
    route_after_model,
    route_after_validation,
    validate_final_node,
)


def _result(name: str, status: str = "ok", call_id: str | None = None) -> dict:
    return {
        "call_id": call_id or name,
        "sub_agent_name": name,
        "status": status,
        "response_text": {
            "materials_database": (
                "| mp-1 | Fe2O3 |\nMATERIAL_QUERY_HANDOFF: query_id=query-test"
            ),
            "data_analysis": "带隙统计：样本数1000，均值2.1。",
            "literature": "Fe2O3 文献题名线索，尚未核验全文。",
        }[name]
        if status == "ok"
        else "",
        "error": {"code": "SUB_AGENT_FAILED", "message": "Intern connection failed"}
        if status == "error"
        else None,
        "warnings": ["这是计算属性"] if name == "materials_database" else [],
    }


def _state(*results: dict, count: int | None = None) -> dict:
    return {
        "user_message": "筛选可见光光催化氧化物，分析带隙并结合文献评估可行性。",
        "executed_call_ids": [result["call_id"] for result in results],
        "sub_agent_results": list(results),
        "sub_agent_call_count": len(results) if count is None else count,
        "model_call_count": 2,
    }


def _context(limit: int = 3) -> SimpleNamespace:
    coordinator = Mock()
    coordinator.material_query_candidates.return_value = (
        {"material_id": "mp-1", "formula_pretty": "Fe2O3"},
    )
    coordinator.material_query_to_analysis.return_value = SimpleNamespace(
        partition=SimpleNamespace(dataset_id="dataset-test")
    )
    return SimpleNamespace(
        cancel_event=None,
        clock=lambda: datetime.now(UTC),
        settings=SimpleNamespace(agent_max_tool_calls_per_turn=limit),
        id_generator=SimpleNamespace(new_id=lambda: "next-call"),
        sub_agent_registry=SimpleNamespace(
            resolve_by_delegate_function=lambda name: SimpleNamespace(
                delegate_function_name=name
            )
        ),
        data_analysis_coordinator=coordinator,
        master_model=SimpleNamespace(
            generate=Mock(side_effect=AssertionError("No remote Master call expected"))
        ),
    )


def test_failed_analysis_continues_literature_without_retrying_analysis() -> None:
    state = _state(_result("materials_database"), _result("data_analysis", "error"))
    context = _context()
    update = call_master_model_node(state, SimpleNamespace(context=context))
    call = update["pending_tool_calls"][0]
    assert call["name"] == "delegate_to_literature"
    task = json.loads(call["arguments_json"])["task"]
    assert state["user_message"] in task and "Fe2O3" in task
    assert "均值2.1" not in task
    context.data_analysis_coordinator.material_query_to_analysis.assert_not_called()
    context.master_model.generate.assert_not_called()
    assert state["sub_agent_call_count"] == 2  # Never reset the budget.


@pytest.mark.parametrize("literature_status", ["ok", "error"])
def test_partial_report_keeps_success_and_failure_after_literature(
    literature_status: str,
) -> None:
    state = _state(
        _result("materials_database"),
        _result("data_analysis", "error"),
        _result("literature", literature_status),
    )
    context = _context()
    update = call_master_model_node(state, SimpleNamespace(context=context))
    assert update["pending_tool_calls"] == []
    draft = AgentFinalDraft.model_validate(update["final_draft"])
    assert draft.status.value == "error"
    assert "mp-1" in draft.answer and "query-test" in draft.answer
    assert "Intern connection failed" in draft.answer
    assert "执行失败" in draft.answer
    assert "均值2.1" not in draft.answer
    assert "不是完整链路成功报告" in draft.answer
    assert draft.evidence_ids == ["materials_database"] + (
        ["literature"] if literature_status == "ok" else []
    )
    assert "这是计算属性" in draft.warnings
    assert draft.follow_up_question is None  # Do not ask for an invented PDF list.
    context.master_model.generate.assert_not_called()


@pytest.mark.parametrize("analysis_status", ["ok", "error"])
def test_exhausted_budget_returns_report_instead_of_another_delegation(
    analysis_status: str,
) -> None:
    state = _state(
        _result("materials_database"),
        _result("data_analysis", analysis_status),
        count=3,
    )
    context = _context()
    update = call_master_model_node(state, SimpleNamespace(context=context))
    assert update["pending_tool_calls"] == []
    draft = update["final_draft"]
    assert draft["status"] == "error"
    assert "文献检索：未执行" in draft["answer"]
    assert "3/3" in draft["answer"]
    assert "mp-1" in draft["answer"]
    assert ("均值2.1" in draft["answer"]) == (analysis_status == "ok")
    assert "PDF" not in (draft["follow_up_question"] or "")
    context.master_model.generate.assert_not_called()
    assert state["sub_agent_call_count"] == 3


def test_database_failure_does_not_launch_dependent_steps_or_retry() -> None:
    state = _state(_result("materials_database", "error"))
    context = _context()
    update = call_master_model_node(state, SimpleNamespace(context=context))
    assert update["pending_tool_calls"] == []
    draft = update["final_draft"]
    assert draft["status"] == "error"
    assert "数据库初筛：执行失败" in draft["answer"]
    assert "数据分析：未执行" in draft["answer"]
    assert "文献检索：未执行" in draft["answer"]
    assert draft["evidence_ids"] == []
    context.master_model.generate.assert_not_called()


def test_previous_turn_failure_does_not_block_current_analysis() -> None:
    state = _state(_result("materials_database"))
    state["sub_agent_results"].append(_result("data_analysis", "error", "old-da"))
    context = _context()
    update = call_master_model_node(state, SimpleNamespace(context=context))
    assert update["pending_tool_calls"][0]["name"] == "delegate_to_data_analysis"


def test_recovered_analysis_uses_success_without_a_further_retry() -> None:
    state = _state(
        _result("materials_database"),
        _result("data_analysis", "error", "failed-da"),
        _result("data_analysis", "ok", "recovered-da"),
    )
    context = _context(limit=4)
    update = call_master_model_node(state, SimpleNamespace(context=context))
    assert update["pending_tool_calls"][0]["name"] == "delegate_to_literature"
    context.data_analysis_coordinator.material_query_to_analysis.assert_not_called()


def test_successful_chain_still_uses_the_normal_full_report() -> None:
    state = _state(
        *(
            _result(name)
            for name in ("materials_database", "data_analysis", "literature")
        )
    )
    context = _context()
    update = call_master_model_node(state, SimpleNamespace(context=context))
    assert update["final_draft"]["status"] == "needs_user_input"
    assert "不是完整链路成功报告" not in update["final_draft"]["answer"]
    assert "均值2.1" in update["final_draft"]["answer"]
    context.master_model.generate.assert_not_called()


def test_unavailable_coordinator_keeps_existing_results_in_partial_report() -> None:
    state = _state(_result("materials_database"))
    context = _context()
    context.data_analysis_coordinator = None
    update = call_master_model_node(state, SimpleNamespace(context=context))
    assert update["pending_tool_calls"] == []
    assert update["final_draft"]["status"] == "error"
    assert "mp-1" in update["final_draft"]["answer"]
    assert "交接" in update["final_draft"]["answer"]
    context.master_model.generate.assert_not_called()


def test_partial_answer_survives_normal_validation_and_finalization() -> None:
    state = _state(
        _result("materials_database"), _result("data_analysis", "error"), count=3
    )
    context = _context()
    runtime = SimpleNamespace(context=context)
    state.update(call_master_model_node(state, runtime))
    assert route_after_model(state) == "validate_final"
    state.update(validate_final_node(state, runtime))
    assert route_after_validation(state) == "finalize_success"
    final = finalize_success_node(state, runtime)
    assert final["final_response"] == state["final_draft"]["answer"]
    assert state["final_draft"]["status"] == "error"
    assert "文献检索：未执行" in final["final_response"]
    assert state["sub_agent_call_count"] == 3


@pytest.mark.parametrize("application", ["可见光光催化", "紫外光电探测"])
def test_failed_literature_is_not_retried_or_marked_successful(
    application: str,
) -> None:
    state = _state(
        _result("materials_database"),
        _result("data_analysis"),
        _result("literature", "error"),
    )
    state["user_message"] = (
        f"筛选用于{application}的氧化物，分析带隙分布并结合文献评估可行性。"
    )
    context = _context(limit=4)
    context.data_analysis_coordinator = None
    update = call_master_model_node(state, SimpleNamespace(context=context))
    assert update["pending_tool_calls"] == []
    assert update["final_draft"]["status"] == "error"
    assert "执行失败" in update["final_draft"]["answer"]
    context.master_model.generate.assert_not_called()


def test_no_delegation_is_queued_when_budget_is_zero() -> None:
    state = _state()
    context = _context(limit=0)
    update = call_master_model_node(state, SimpleNamespace(context=context))
    assert update["pending_tool_calls"] == []
    assert update["final_draft"]["status"] == "error"
    assert "0/0" in update["final_draft"]["answer"]
    assert "数据库初筛：未执行" in update["final_draft"]["answer"]
    context.master_model.generate.assert_not_called()
