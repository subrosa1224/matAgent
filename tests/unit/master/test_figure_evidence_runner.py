"""Actual Master ask/checkpointer with offline proposals, never remote science."""

import sqlite3
from unittest.mock import Mock

from langgraph.checkpoint.sqlite import SqliteSaver

from materials_screening.agent.conversation_store import SqliteConversationStore
from materials_screening.master.figure_evidence_review import FigureEvidenceService
from materials_screening.master.figure_evidence_store import FigureEvidenceStore
from tests.unit.master.test_fulltext_analysis import analysis_env
from tests.unit.master.test_fulltext_task_state import make_runner, seed


def test_actual_master_continue_cannot_approve_and_closed_restores_once(tmp_path):
    def configure(env):
        env.task = env.task.model_copy(
            update={"figure_review_policy": "figure-evidence-review-v1"}
        )
        env.service = FigureEvidenceService(
            FigureEvidenceStore(tmp_path / "figures"),
            env.processor.artifacts,
            env.snapshots,
        )
        env.processor.extraction_processor.figure_review_service = env.service

    env = analysis_env(tmp_path, configure=configure)
    env.processor.extraction_processor.analysis_processor.figure_review_service = (
        env.service
    )
    store = SqliteConversationStore(tmp_path / "conversations.sqlite")
    model = Mock()
    with sqlite3.connect(
        tmp_path / "checkpoints.sqlite", check_same_thread=False
    ) as connection:
        runner = make_runner(
            tmp_path, store, SqliteSaver(connection), model, env.processor.artifacts
        )
        runner._figure_review_service = env.service
        runner._preview_processor = env.processor
        seed(runner, store, task=env.task)
        for message in ["继续", "确定", "都正确"]:
            result = runner.ask(
                message=message, conversation_id="conv-a", task_id=env.task.task_id
            )
            assert (
                result.final_status == "needs_user_input"
                and result.model_call_count == 0
            )
            saved = runner.get_fulltext_tasks("conv-a")[0]
            assert saved == env.task
        model.generate.assert_not_called()
        assert env.scope_calls == 0
        runner.figure_review_action(
            conversation_id="conv-a",
            task_id=env.task.task_id,
            expected=env.task.figure_evidence_ref,
            operation_id="explicit-skip",
            action="close",
            abandon_all=True,
        )
        result = runner.ask(
            message="继续", conversation_id="conv-a", task_id=env.task.task_id
        )
        assert result.status == "completed", result.model_dump()
        finished = runner.get_fulltext_tasks("conv-a")[0]
        assert finished.stage == "finished" and result.tool_call_count == 2
        assert "实验条件尚未绑定" in result.response_text
        snapshots = tuple((tmp_path / "snapshots").glob("*.json"))
        repeat = runner.ask(
            message="继续", conversation_id="conv-a", task_id=env.task.task_id
        )
        assert repeat.model_call_count == repeat.tool_call_count == 0
        assert env.scope_calls == 1
        assert tuple((tmp_path / "snapshots").glob("*.json")) == snapshots
        assert runner.get_fulltext_tasks("conv-a")[0] == finished
    store.close()


def test_new_policy_flag_never_upgrades_existing_task(tmp_path):
    from materials_screening.agent.models import AgentResult
    from tests.unit.master.test_fulltext_task_state import QUESTION

    store = SqliteConversationStore(tmp_path / "conversations.sqlite")
    with sqlite3.connect(
        tmp_path / "checkpoints.sqlite", check_same_thread=False
    ) as connection:
        runner = make_runner(tmp_path, store, SqliteSaver(connection), Mock(), None)
        old = seed(runner, store)
        runner._figure_review_enabled = True
        assert runner.get_fulltext_tasks("conv-a")[0].figure_review_policy == "disabled"
        new = runner._remember_fulltext_task(
            QUESTION,
            AgentResult(
                conversation_id="conv-a",
                user_turn_id="unit",
                status="completed",
                final_status="needs_user_input",
                response_text="provide PDF",
            ),
        )
        assert new.figure_review_policy == "figure-evidence-review-v1"
        assert (
            next(
                t
                for t in runner.get_fulltext_tasks("conv-a")
                if t.task_id == old.task_id
            )
            == old
        )
    store.close()
