import sqlite3
from unittest.mock import Mock

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

from materials_screening.agent.conversation_store import SqliteConversationStore
from materials_screening.master.preview_retry_ui import (
    PreviewRetryController,
    mount_preview_retry,
)
from tests.unit.master.test_fulltext_preview import failed_preview, setup
from tests.unit.master.test_fulltext_task_state import make_runner, seed


def test_actual_runner_and_ui_retry_are_scoped_idempotent_and_not_delegated(tmp_path):
    env = setup(tmp_path)
    old = failed_preview(env)
    store = SqliteConversationStore(tmp_path / "conversations.sqlite")
    model = Mock()
    with sqlite3.connect(
        tmp_path / "checkpoint.sqlite", check_same_thread=False
    ) as connection:
        runner = make_runner(
            tmp_path, store, SqliteSaver(connection), model, env.processor.artifacts
        )
        runner._preview_processor = env.processor
        seed(runner, store, task=old)
        controller = PreviewRetryController(lambda: runner)
        session = {"master_conversation_id": "conv-a"}
        scope, text = controller.open(session, old.task_id, old.document_ids[0])
        assert "0/2" in text
        with pytest.raises(ValueError, match="会话"):
            controller.retry({"master_conversation_id": "other"}, scope)
        response = controller.retry(session, scope)
        assert "1/2" in response and "succeeded" in response
        saved = runner.get_fulltext_tasks("conv-a")[0]
        assert saved.preview_retry_history[0].previous_preview == old.previews[0]
        controller.retry(session, scope)
        assert env.llm.calls == 2
        result = runner.ask(
            message="继续", conversation_id="conv-a", task_id=old.task_id
        )
        assert result.model_call_count == 0
        assert (
            runner.get_fulltext_tasks("conv-a")[0]
            .preview_decisions[old.document_ids[0]]
            .action
            == "hold"
        )
        model.generate.assert_not_called()
    store.close()


def test_mount_preview_retry_builds_real_gradio_panel():
    import gradio as gr

    with gr.Blocks() as app:
        state = gr.State({})
        refresh, outputs = mount_preview_retry(state, lambda: None)
    assert len(outputs) == 5
    assert len(refresh({})) == 5
    labels = [c.get("props", {}).get("value") for c in app.config["components"]]
    assert "重新生成预览" in labels
