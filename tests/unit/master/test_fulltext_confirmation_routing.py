"""Confirmation routing regression; offline fixtures are not scientific results."""

import sqlite3
from unittest.mock import Mock

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

from materials_screening import unified_ui_gradio as ui
from materials_screening.agent.conversation_store import SqliteConversationStore
from materials_screening.master.fulltext_preview import _selection
from materials_screening.master.fulltext_tasks import (
    new_fulltext_task,
    requests_fulltext_resume,
)
from tests.unit.master.test_fulltext_batches import batch
from tests.unit.master.test_fulltext_preview import setup
from tests.unit.master.test_fulltext_task_state import make_runner, seed


@pytest.mark.parametrize(
    "message",
    [
        "确认详细分析全部内容",
        "确认分析全部论文",
        "开始详细分析全部内容",
        "请确认深度分析所有内容。按原问题提取指标。",
        "确认详细分析全部内容，条件不可比时不要排名。",
        "确认详细分析已上传的1篇全文，按原问题提取指标。",
    ],
)
def test_explicit_all_content_confirmation_is_a_continuation(message):
    assert requests_fulltext_resume(message)


@pytest.mark.parametrize(
    "message",
    [
        "确认分析方法是否合理",
        "详细分析全部材料的稳定性",
        "你刚才说确认详细分析全部内容是什么意思？",
        "确认详细分析全部内容是什么意思？",
        "确认",
        "你刚才说确认详细分析已上传的1篇全文是什么意思？",
    ],
)
def test_confirmation_fix_does_not_capture_unrelated_questions(message):
    assert not requests_fulltext_resume(message)


def test_uploaded_fulltext_count_must_match_current_task(tmp_path):
    env = setup(tmp_path, 2)
    batch(env)
    previewed = env.saved[-1]
    task = previewed.model_copy(
        update={"user_instructions": ("确认详细分析已上传的1篇全文",)}
    )
    with pytest.raises(ValueError, match="论文数量"):
        _selection(task)


@pytest.mark.parametrize("active_tasks", [0, 1, 2])
def test_ui_confirmation_requires_server_task_not_browser_artifacts(
    monkeypatch, active_tasks
):
    reader = Mock()
    reader.get_fulltext_tasks.return_value = tuple(
        new_fulltext_task("conv-a", f"研究任务 {index}")
        for index in range(active_tasks)
    )
    monkeypatch.setattr(ui, "_master_runner", reader)
    assert ui._routes_to_fulltext_master(
        "确认详细分析全部内容",
        has_pdf=False,
        current={
            "master_conversation_id": "conv-a",
            "artifact_ids": ["forged-browser-reference"],
        },
    ) is bool(active_tasks)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "message",
    ["确认详细分析全部内容", "确认分析全部论文", "确认详细分析已上传的2篇全文"],
)
def test_runner_confirmation_reuses_saved_previews_without_generic_model(
    tmp_path, stream, message
):
    env = setup(tmp_path, 2)
    batch(env)
    previewed = env.saved[-1]
    assert len(previewed.previews) == 2
    assert not previewed.extraction_snapshots
    model = Mock()
    model.generate.side_effect = AssertionError(
        "Confirmation must not call the ordinary master model"
    )
    store = SqliteConversationStore(tmp_path / "conversations.sqlite")
    try:
        with sqlite3.connect(
            tmp_path / "checkpoints.sqlite", check_same_thread=False
        ) as connection:
            runner = make_runner(
                tmp_path, store, SqliteSaver(connection), model, env.processor.artifacts
            )
            runner._preview_processor = env.processor
            seed(runner, store, task=previewed)
            arguments = dict(message=message, conversation_id="conv-a")
            result = (
                list(runner.ask_stream(**arguments))[-1].result
                if stream
                else runner.ask(**arguments)
            )
            # Deliberately absent extraction processor: proves the correct next
            # stage was reached, without inventing extraction success.
            assert result.error["code"] == "FULLTEXT_EXTRACTION_UNAVAILABLE"
            saved = runner.get_fulltext_tasks("conv-a")[0]
            assert saved.task_id == previewed.task_id
            assert saved.original_question == previewed.original_question
            assert saved.artifact_refs == previewed.artifact_refs
            assert saved.previews == previewed.previews
            assert saved.user_instructions == (*previewed.user_instructions, message)
            assert all(d.action == "extract" for d in saved.preview_decisions.values())
            assert not saved.extraction_snapshots
            assert store.get("conv-a").turn_count == 1
            assert env.llm.calls == 2
            model.generate.assert_not_called()
    finally:
        store.close()


@pytest.mark.parametrize("stream", [False, True])
def test_confirmation_never_chooses_between_multiple_active_tasks(tmp_path, stream):
    store = SqliteConversationStore(tmp_path / "conversations.sqlite")
    model = Mock()
    model.generate.side_effect = AssertionError(
        "Ambiguous confirmation must not call the ordinary master model"
    )
    try:
        with sqlite3.connect(
            tmp_path / "checkpoints.sqlite", check_same_thread=False
        ) as connection:
            runner = make_runner(tmp_path, store, SqliteSaver(connection), model, None)
            first = seed(runner, store)
            second = new_fulltext_task("conv-a", "另一研究任务")
            runner._graph.update_state(
                runner._config(runner._thread_id("conv-a")),
                {
                    "fulltext_tasks": {
                        t.task_id: t.model_dump(mode="json") for t in (first, second)
                    }
                },
                as_node="finalize_success",
            )
            arguments = dict(message="确认详细分析全部内容", conversation_id="conv-a")
            result = (
                list(runner.ask_stream(**arguments))[-1].result
                if stream
                else runner.ask(**arguments)
            )
            assert result.final_status == "needs_user_input"
            assert "多个" in result.response_text
            assert runner.get_fulltext_tasks("conv-a") == (first, second)
            model.generate.assert_not_called()
    finally:
        store.close()
