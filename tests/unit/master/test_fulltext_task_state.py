"""Persistent task binding; no test substitutes a real extraction success."""

import json
import sqlite3
from pathlib import Path
from unittest.mock import Mock

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

from materials_screening.agent.conversation_store import SqliteConversationStore
from materials_screening.master.artifact_registry import ArtifactRegistry
from materials_screening.master.fulltext_tasks import (
    FulltextTask,
    FulltextTaskSelectionError,
    new_fulltext_task,
    requests_fulltext_task,
    select_fulltext_task,
)
from materials_screening.master.master_runner import MasterAgentRunner
from materials_screening.master.master_settings import MasterAgentSettings
from materials_screening.master.sub_agent_registry import SubAgentRegistry

QUESTION = (
    "查询GdVO4材料并分析稳定性；检索复合光催化实验论文，"
    "上传全文后提取降解率并统计变化。"
)


def test_task_keeps_question_and_only_current_turn_references() -> None:
    state = {
        "executed_call_ids": ["current"],
        "sub_agent_results": [
            {"call_id": "old", "response_text": "query-" + "a" * 32},
            {
                "call_id": "current",
                "response_text": "query-" + "b" * 32 + " dataset-" + "c" * 24,
            },
        ],
    }
    task = new_fulltext_task("conv-a", QUESTION, state=state)
    assert task.original_question == QUESTION
    assert task.database_query_ids == ("query-" + "b" * 32,)
    assert task.database_dataset_ids == ("dataset-" + "c" * 24,)
    assert task.analysis_mode == "trial"
    assert task.stage == "waiting_fulltext"
    assert FulltextTask.model_validate_json(task.model_dump_json()) == task


def test_selection_rejects_cross_conversation_or_ambiguous_tasks() -> None:
    first = new_fulltext_task("conv-a", QUESTION)
    second = new_fulltext_task("conv-a", QUESTION + "同时比较另一批论文。")
    with pytest.raises(FulltextTaskSelectionError, match="多个"):
        select_fulltext_task((first, second), conversation_id="conv-a")
    with pytest.raises(FulltextTaskSelectionError, match="会话"):
        select_fulltext_task((first,), conversation_id="conv-b", task_id=first.task_id)
    assert (
        select_fulltext_task(
            (first, second), conversation_id="conv-a", task_id=second.task_id
        )
        == second
    )


@pytest.mark.parametrize("question,has_task", [(QUESTION, True), ("你好", False)])
def test_real_master_graph_checkpoint_keeps_optional_attachment_compatibility(
    tmp_path, question, has_task
):
    """Real graph/checkpointer with offline model, not a scientific-chain run."""
    from materials_screening.agent.model_base import (
        AgentModelStatus,
        MaterialAgentResponse,
    )
    from materials_screening.agent.models import AgentMessageItem

    model = Mock()
    model.generate.return_value = MaterialAgentResponse(
        status=AgentModelStatus.COMPLETED,
        provider="unit",
        model="offline-unit",
        request_id="unit-request",
        output_items=(
            AgentMessageItem(
                role="assistant",
                content=json.dumps(
                    {
                        "status": "needs_user_input" if has_task else "completed",
                        "answer": "测试回答",
                        "referenced_material_ids": [],
                        "evidence_ids": [],
                        "warnings": [],
                        "follow_up_question": "请提供PDF" if has_task else None,
                    },
                    ensure_ascii=False,
                ),
            ),
        ),
    )
    store = SqliteConversationStore(tmp_path / "conversations.sqlite")
    with sqlite3.connect(
        tmp_path / "checkpoints.sqlite", check_same_thread=False
    ) as connection:
        runner = make_runner(
            tmp_path,
            store,
            SqliteSaver(connection),
            model,
            ArtifactRegistry(tmp_path / "artifacts"),
        )
        result = runner.ask(message=question)
        assert result.status == "completed", result.model_dump()
        assert result.response_text == "测试回答"
        assert model.generate.call_count == 1
        assert store.get(result.conversation_id).turn_count == 1
        tasks = runner.get_fulltext_tasks(result.conversation_id)
        assert bool(tasks) == has_task
        if has_task:
            assert tasks[0].original_question == question
            assert tasks[0].stage == "waiting_fulltext"
    store.close()


def test_finished_task_not_selected_by_continue() -> None:
    task = new_fulltext_task("conv-a", QUESTION)
    finished = task.model_copy(update={"stage": "finished"})
    assert select_fulltext_task((finished,), conversation_id="conv-a") is None


def make_runner(tmp_path: Path, store, saver, model, registry) -> MasterAgentRunner:
    return MasterAgentRunner(
        settings=MasterAgentSettings(),
        store=store,
        sub_agent_registry=SubAgentRegistry(),
        master_model=model,
        checkpointer=saver,
        artifact_registry=registry,
    )


def seed(runner, store, *, conversation="conv-a", task=None) -> FulltextTask:
    store.create(conversation)
    task = task or new_fulltext_task(conversation, QUESTION)
    runner._graph.update_state(
        {"configurable": {"thread_id": "master_" + conversation}},
        {
            "conversation_id": conversation,
            "user_message": QUESTION,
            "fulltext_tasks": {task.task_id: task.model_dump(mode="json")},
            "status": "completed",
            "final_draft": {"status": "needs_user_input"},
        },
        as_node="finalize_success",
    )
    return task


def register_pdf(tmp_path, registry, conversation="conv-a"):
    path = tmp_path / "paper.pdf"
    path.write_bytes(b"%PDF-1.7\nsource")
    return registry.register_pdf(path, conversation_id=conversation)


def test_upload_binds_existing_task_and_survives_sqlite_restart(tmp_path: Path) -> None:
    model = Mock()
    registry = ArtifactRegistry(tmp_path / "artifacts")
    store = SqliteConversationStore(tmp_path / "conversations.sqlite")
    with sqlite3.connect(
        tmp_path / "checkpoints.sqlite", check_same_thread=False
    ) as connection:
        runner = make_runner(tmp_path, store, SqliteSaver(connection), model, registry)
        task = seed(runner, store)
        artifact = register_pdf(tmp_path, registry)
        result = runner.ask(
            message="", conversation_id="conv-a", artifact_refs=(artifact.artifact_id,)
        )
        assert result.error["code"] == "FULLTEXT_PROCESSOR_UNAVAILABLE"
        saved = runner.get_fulltext_tasks("conv-a")[0]
        assert saved.original_question == QUESTION
        assert saved.artifact_refs == (artifact.artifact_id,)
        assert saved.document_ids == ("doc-" + artifact.metadata["sha256"][:24],)
        assert saved.stage == "ready_to_resume"
        assert saved.resume_stage == "previewing"
        assert saved.task_id == task.task_id
        model.generate.assert_not_called()
    store.close()
    reopened = SqliteConversationStore(tmp_path / "conversations.sqlite")
    with sqlite3.connect(
        tmp_path / "checkpoints.sqlite", check_same_thread=False
    ) as connection:
        runner = make_runner(
            tmp_path,
            reopened,
            SqliteSaver(connection),
            model,
            ArtifactRegistry(tmp_path / "artifacts"),
        )
        assert runner.get_fulltext_tasks("conv-a")[0] == saved
        result = runner.ask(message="继续", conversation_id="conv-a")
        assert result.error["code"] == "FULLTEXT_PROCESSOR_UNAVAILABLE"
        assert (
            runner.get_fulltext_tasks("conv-a")[0].artifact_refs == saved.artifact_refs
        )
        model.generate.assert_not_called()
    reopened.close()


@pytest.mark.parametrize("stream", [False, True])
def test_initial_attachment_preserves_first_stage_result_and_counts_one_turn(
    tmp_path, monkeypatch, stream
):
    """First-stage substitute tests ordering only, not actual database science."""
    from materials_screening.agent.models import AgentResult
    from materials_screening.master.master_runner import AgentStreamEvent

    registry = ArtifactRegistry(tmp_path / "artifacts")
    store = SqliteConversationStore(tmp_path / "conversations.sqlite")
    with sqlite3.connect(
        tmp_path / "checkpoints.sqlite", check_same_thread=False
    ) as connection:
        runner = make_runner(tmp_path, store, SqliteSaver(connection), Mock(), registry)
        store.create("conv-a")
        runner._settings = runner._settings.model_copy(
            update={"master_max_conversation_turns": 1}
        )
        artifact = register_pdf(tmp_path, registry)
        stages = []

        def first_stage(**kwargs):
            assert kwargs["message"] == QUESTION
            assert runner.get_fulltext_tasks("conv-a") == ()
            stages.append("initial")
            store.record_turn("conv-a")
            return AgentResult(
                conversation_id="conv-a",
                user_turn_id="initial-turn",
                status="completed",
                final_status="needs_user_input",
                response_text="第一阶段结果（测试替身）",
                model_call_count=2,
                tool_call_count=3,
                selected_tools=("delegate_to_materials_database",),
                evidence_ids=("db-call",),
            )

        def first_stream(**kwargs):
            yield AgentStreamEvent(
                node="runner",
                message="first",
                is_final=True,
                result=first_stage(message=kwargs["message"]),
            )

        monkeypatch.setattr(runner, "_ask_locked", first_stage)
        monkeypatch.setattr(runner, "_ask_stream_locked", first_stream)
        kwargs = dict(
            message=QUESTION,
            conversation_id="conv-a",
            artifact_refs=(artifact.artifact_id,),
        )
        result = (
            list(runner.ask_stream(**kwargs))[-1].result
            if stream
            else runner.ask(**kwargs)
        )
        assert stages == ["initial"]
        assert result.error["code"] == "FULLTEXT_PROCESSOR_UNAVAILABLE"
        assert "第一阶段结果" in result.response_text
        assert result.model_call_count == 2
        assert result.tool_call_count == 3
        assert result.evidence_ids == ("db-call",)
        assert store.get("conv-a").turn_count == 1
        assert runner.get_fulltext_tasks("conv-a")[0].artifact_refs == (
            artifact.artifact_id,
        )
    store.close()


@pytest.mark.parametrize("message", ["只预览这些论文", "只分析第2篇"])
def test_followup_saved_without_overwriting_original_question(tmp_path, message):
    registry = ArtifactRegistry(tmp_path / "artifacts")
    store = SqliteConversationStore(tmp_path / "conversations.sqlite")
    with sqlite3.connect(
        tmp_path / "checkpoints.sqlite", check_same_thread=False
    ) as connection:
        model = Mock()
        runner = make_runner(tmp_path, store, SqliteSaver(connection), model, registry)
        seed(runner, store)
        artifact = register_pdf(tmp_path, registry)
        runner.ask(
            message="继续",
            conversation_id="conv-a",
            artifact_refs=(artifact.artifact_id,),
        )
        result = runner.ask(message=message, conversation_id="conv-a")
        assert result.error["code"] == "FULLTEXT_PROCESSOR_UNAVAILABLE"
        task = runner.get_fulltext_tasks("conv-a")[0]
        assert task.original_question == QUESTION
        assert task.user_instructions == (message,)
        model.generate.assert_not_called()
    store.close()


@pytest.mark.parametrize(
    "question,expected",
    [
        (QUESTION, True),
        ("查询ZnO，分析密度，检索实验论文。", False),
        ("分析上传的PDF。", False),
        ("查询ZnO，分析密度，检索论文；不要提取实验数据。", False),
        ("查询ZnO，分析密度，检索论文；不用全文。", False),
    ],
)
def test_explicit_fulltext_request_not_inferred_from_paper_background(
    question, expected
):
    assert requests_fulltext_task(question) is expected


def test_preview_override_before_upload_is_not_lost(tmp_path):
    registry = ArtifactRegistry(tmp_path / "artifacts")
    store = SqliteConversationStore(tmp_path / "conversations.sqlite")
    with sqlite3.connect(
        tmp_path / "checkpoints.sqlite", check_same_thread=False
    ) as connection:
        model = Mock()
        runner = make_runner(tmp_path, store, SqliteSaver(connection), model, registry)
        seed(runner, store)
        result = runner.ask(message="只预览这些论文", conversation_id="conv-a")
        assert result.final_status == "needs_user_input"
        assert runner.get_fulltext_tasks("conv-a")[0].user_instructions == (
            "只预览这些论文",
        )
        artifact = register_pdf(tmp_path, registry)
        runner.ask(
            message="", conversation_id="conv-a", artifact_refs=(artifact.artifact_id,)
        )
        assert runner.get_fulltext_tasks("conv-a")[0].user_instructions == (
            "只预览这些论文",
        )
        model.generate.assert_not_called()
    store.close()


def test_corrupt_artifact_catalog_returns_safe_error_and_keeps_task(tmp_path):
    registry = ArtifactRegistry(tmp_path / "artifacts")
    store = SqliteConversationStore(tmp_path / "conversations.sqlite")
    with sqlite3.connect(
        tmp_path / "checkpoints.sqlite", check_same_thread=False
    ) as connection:
        model = Mock()
        runner = make_runner(tmp_path, store, SqliteSaver(connection), model, registry)
        task = seed(runner, store)
        artifact = register_pdf(tmp_path, registry)
        (tmp_path / "artifacts/artifacts.sqlite").write_bytes(b"broken sqlite")
        result = runner.ask(
            message="继续",
            conversation_id="conv-a",
            artifact_refs=(artifact.artifact_id,),
        )
        assert result.error["code"] == "FULLTEXT_ATTACHMENT_REJECTED"
        assert str(tmp_path) not in result.response_text
        assert runner.get_fulltext_tasks("conv-a") == (task,)
        model.generate.assert_not_called()
    store.close()


@pytest.mark.parametrize("message", ["继续", QUESTION])
def test_unauthorized_upload_never_mutates_task(tmp_path: Path, message: str) -> None:
    registry = ArtifactRegistry(tmp_path / "artifacts")
    store = SqliteConversationStore(tmp_path / "conversations.sqlite")
    with sqlite3.connect(
        tmp_path / "checkpoints.sqlite", check_same_thread=False
    ) as connection:
        model = Mock()
        runner = make_runner(tmp_path, store, SqliteSaver(connection), model, registry)
        task = seed(runner, store)
        artifact = register_pdf(tmp_path, registry, "conv-b")
        result = runner.ask(
            message=message,
            conversation_id="conv-a",
            artifact_refs=(artifact.artifact_id,),
        )
        assert result.error["code"] == "FULLTEXT_ATTACHMENT_REJECTED"
        assert runner.get_fulltext_tasks("conv-a") == (task,)
        model.generate.assert_not_called()
    store.close()


def test_multiple_tasks_ask_for_selection_without_consuming_model(
    tmp_path: Path,
) -> None:
    registry = ArtifactRegistry(tmp_path / "artifacts")
    store = SqliteConversationStore(tmp_path / "conversations.sqlite")
    with sqlite3.connect(
        tmp_path / "checkpoints.sqlite", check_same_thread=False
    ) as connection:
        model = Mock()
        runner = make_runner(tmp_path, store, SqliteSaver(connection), model, registry)
        task = seed(runner, store)
        other = new_fulltext_task("conv-a", QUESTION + "另一任务。")
        assert other.task_id != task.task_id
        runner._graph.update_state(
            runner._config(runner._thread_id("conv-a")),
            {
                "fulltext_tasks": {
                    item.task_id: item.model_dump(mode="json") for item in (task, other)
                }
            },
            as_node="finalize_success",
        )
        assert len(runner.get_fulltext_tasks("conv-a")) == 2, runner.get_fulltext_tasks(
            "conv-a"
        )
        artifact = register_pdf(tmp_path, registry)
        result = runner.ask(
            message="继续",
            conversation_id="conv-a",
            artifact_refs=(artifact.artifact_id,),
        )
        assert result.final_status == "needs_user_input", result.model_dump()
        assert "多个" in result.response_text
        assert all(
            not item.artifact_refs for item in runner.get_fulltext_tasks("conv-a")
        )
        model.generate.assert_not_called()
    store.close()


def test_legacy_waiting_conversation_restores_task_from_server_checkpoint(
    tmp_path: Path,
) -> None:
    registry = ArtifactRegistry(tmp_path / "artifacts")
    store = SqliteConversationStore(tmp_path / "conversations.sqlite")
    with sqlite3.connect(
        tmp_path / "checkpoints.sqlite", check_same_thread=False
    ) as connection:
        model = Mock()
        runner = make_runner(tmp_path, store, SqliteSaver(connection), model, registry)
        seed(runner, store)
        runner._graph.update_state(
            runner._config(runner._thread_id("conv-a")),
            {"fulltext_tasks": {}},
            as_node="finalize_success",
        )
        assert (
            runner._graph.get_state(runner._config(runner._thread_id("conv-a"))).values[
                "fulltext_tasks"
            ]
            == {}
        )
        artifact = register_pdf(tmp_path, registry)
        result = runner.ask(
            message="继续",
            conversation_id="conv-a",
            artifact_refs=(artifact.artifact_id,),
        )
        assert result.error["code"] == "FULLTEXT_PROCESSOR_UNAVAILABLE"
        assert runner.get_fulltext_tasks("conv-a")[0].original_question == QUESTION
        model.generate.assert_not_called()
    store.close()


def test_corrupt_task_state_is_visible_and_not_replaced(tmp_path: Path) -> None:
    registry = ArtifactRegistry(tmp_path / "artifacts")
    store = SqliteConversationStore(tmp_path / "conversations.sqlite")
    with sqlite3.connect(
        tmp_path / "checkpoints.sqlite", check_same_thread=False
    ) as connection:
        model = Mock()
        runner = make_runner(tmp_path, store, SqliteSaver(connection), model, registry)
        seed(runner, store)
        invalid = {"bad-task": {"original_question": "forged"}}
        runner._graph.update_state(
            runner._config(runner._thread_id("conv-a")),
            {"fulltext_tasks": invalid},
            as_node="finalize_success",
        )
        result = runner.ask(message="继续", conversation_id="conv-a")
        assert result.error["code"] == "FULLTEXT_TASK_STATE_INVALID"
        assert (
            runner._graph.get_state(runner._config(runner._thread_id("conv-a"))).values[
                "fulltext_tasks"
            ]
            == invalid
        )
        model.generate.assert_not_called()
    store.close()


def test_cancelled_attachment_request_has_no_task_mutation(tmp_path: Path) -> None:
    import threading

    registry = ArtifactRegistry(tmp_path / "artifacts")
    store = SqliteConversationStore(tmp_path / "conversations.sqlite")
    with sqlite3.connect(
        tmp_path / "checkpoints.sqlite", check_same_thread=False
    ) as connection:
        model = Mock()
        runner = make_runner(tmp_path, store, SqliteSaver(connection), model, registry)
        task = seed(runner, store)
        artifact = register_pdf(tmp_path, registry)
        cancelled = threading.Event()
        cancelled.set()
        events = list(
            runner.ask_stream(
                message="继续",
                conversation_id="conv-a",
                artifact_refs=(artifact.artifact_id,),
                cancel_event=cancelled,
            )
        )
        assert events[-1].result.status == "cancelled"
        assert runner.get_fulltext_tasks("conv-a") == (task,)
        model.generate.assert_not_called()
    store.close()


def test_stream_uses_same_binding_and_never_returns_success_for_missing_processor(
    tmp_path: Path,
) -> None:
    registry = ArtifactRegistry(tmp_path / "artifacts")
    store = SqliteConversationStore(tmp_path / "conversations.sqlite")
    with sqlite3.connect(
        tmp_path / "checkpoints.sqlite", check_same_thread=False
    ) as connection:
        model = Mock()
        runner = make_runner(tmp_path, store, SqliteSaver(connection), model, registry)
        seed(runner, store)
        artifact = register_pdf(tmp_path, registry)
        events = list(
            runner.ask_stream(
                message="继续",
                conversation_id="conv-a",
                artifact_refs=(artifact.artifact_id,),
            )
        )
        assert events[-1].is_final
        assert events[-1].result.error["code"] == "FULLTEXT_PROCESSOR_UNAVAILABLE"
        model.generate.assert_not_called()
    store.close()
