"""Recovery binds durable tasks only; it never invokes a scientific workflow."""

import sqlite3
from unittest.mock import Mock

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

from materials_screening import unified_ui_gradio as ui
from materials_screening.agent.conversation_store import SqliteConversationStore
from materials_screening.master.fulltext_tasks import new_fulltext_task
from tests.unit.master.test_fulltext_preview import run, setup
from tests.unit.master.test_fulltext_task_state import make_runner, seed


def test_restore_ten_pdfs_survives_restart_without_changing_task_or_turn(
    tmp_path, monkeypatch
):
    env = setup(tmp_path, 10)
    model = Mock()
    store = SqliteConversationStore(tmp_path / "conversations.sqlite")
    try:
        with sqlite3.connect(
            tmp_path / "checkpoints.sqlite", check_same_thread=False
        ) as connection:
            runner = make_runner(
                tmp_path, store, SqliteSaver(connection), model, env.processor.artifacts
            )
            seed(runner, store, task=env.task)
        with sqlite3.connect(
            tmp_path / "checkpoints.sqlite", check_same_thread=False
        ) as connection:
            runner = make_runner(
                tmp_path, store, SqliteSaver(connection), model, env.processor.artifacts
            )
            monkeypatch.setattr(ui, "_master_runner", runner)
            payload = ui._restore_conversation_ui("conv-a", False)
            assert len(payload) == len(ui._new_conversation_ui(False))
            assert payload[2]["master_conversation_id"] == "conv-a"
            assert payload[2]["artifact_ids"] == list(env.task.artifact_refs)
            assert len(payload[2]["artifact_names"]) == 10
            assert payload[0][0]["content"] == env.task.original_question
            assert "10" in payload[0][-1]["content"]
            assert "不是" in payload[0][-1]["content"]
            assert runner.get_fulltext_tasks("conv-a") == (env.task,)
            assert store.get("conv-a").turn_count == 0
            model.generate.assert_not_called()
    finally:
        store.close()


def test_restore_rejects_ambiguous_task_and_foreign_grant(tmp_path):
    env = setup(tmp_path)
    model = Mock()
    store = SqliteConversationStore(tmp_path / "conversations.sqlite")
    try:
        with sqlite3.connect(
            tmp_path / "checkpoints.sqlite", check_same_thread=False
        ) as connection:
            runner = make_runner(
                tmp_path, store, SqliteSaver(connection), model, env.processor.artifacts
            )
            seed(runner, store, task=env.task)
            runner._save_fulltext_task(new_fulltext_task("conv-a", "另一个全文任务"))
            with pytest.raises(ValueError, match="多个"):
                runner.restore_fulltext_context("conv-a")
            foreign = env.task.model_copy(update={"conversation_id": "conv-b"})
            seed(runner, store, conversation="conv-b", task=foreign)
            with pytest.raises(ValueError, match="授权"):
                runner.restore_fulltext_context("conv-b")
            model.generate.assert_not_called()
    finally:
        store.close()


def test_restore_does_not_switch_a_busy_conversation(tmp_path):
    env = setup(tmp_path)
    store = SqliteConversationStore(tmp_path / "conversations.sqlite")
    try:
        with sqlite3.connect(
            tmp_path / "checkpoints.sqlite", check_same_thread=False
        ) as connection:
            runner = make_runner(
                tmp_path,
                store,
                SqliteSaver(connection),
                Mock(),
                env.processor.artifacts,
            )
            seed(runner, store, task=env.task)
            lock = store.conversation_lock("conv-a")
            lock.acquire()
            try:
                with pytest.raises(ValueError, match="正在处理"):
                    runner.restore_fulltext_context("conv-a")
            finally:
                lock.release()
    finally:
        store.close()


def test_restore_shows_verified_cached_previews_without_model_or_checkpoint_write(
    tmp_path, monkeypatch
):
    env = setup(tmp_path, 2)
    run(env, env.task.model_copy(update={"user_instructions": ("只预览",)}))
    task = env.saved[-1]
    store = SqliteConversationStore(tmp_path / "conversations.sqlite")
    model = Mock()
    try:
        with sqlite3.connect(
            tmp_path / "checkpoints.sqlite", check_same_thread=False
        ) as connection:
            runner = make_runner(
                tmp_path, store, SqliteSaver(connection), model, env.processor.artifacts
            )
            seed(runner, store, task=task)
            monkeypatch.setattr(ui, "_master_runner", runner)
            payload = ui._restore_conversation_ui("conv-a", False)
            main = payload[0][-1]["content"].split("<details>")[0]
            assert "已预览 2/2 篇" in main and "等待你确认分析范围" in main
            assert "第1篇" in main and "第2篇" in main
            assert "hold" not in main and "继续预览" not in main
            assert runner.get_fulltext_tasks("conv-a") == (task,)
            assert store.get("conv-a").turn_count == 0
            model.generate.assert_not_called()
            assert env.llm.calls == 2
    finally:
        store.close()


def test_restore_cannot_display_forged_cached_quote(tmp_path):
    env = setup(tmp_path)
    run(env, env.task.model_copy(update={"user_instructions": ("只预览",)}))
    task = env.saved[-1]
    task = task.model_copy(
        update={
            "previews": (
                task.previews[0].model_copy(update={"evidence_quote": "foreign quote"}),
            )
        }
    )
    store = SqliteConversationStore(tmp_path / "conversations.sqlite")
    try:
        with sqlite3.connect(
            tmp_path / "checkpoints.sqlite", check_same_thread=False
        ) as connection:
            runner = make_runner(
                tmp_path,
                store,
                SqliteSaver(connection),
                Mock(),
                env.processor.artifacts,
            )
            seed(runner, store, task=task)
            with pytest.raises(ValueError, match="源片段不一致"):
                runner.restore_fulltext_context("conv-a")
            assert runner.get_fulltext_tasks("conv-a") == (task,)
    finally:
        store.close()
