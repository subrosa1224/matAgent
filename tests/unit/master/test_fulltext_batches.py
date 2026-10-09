"""Offline batch orchestration; not real-paper or scientific acceptance."""

import threading

import pytest

from tests.unit.master.test_fulltext_preview import setup


def batch(env, **changes):
    env.processor.auto_batches = True
    env.processor.batch_seconds = 600
    env.processor.max_batches = 20
    env.processor.require_preview_confirmation = True
    for key, value in changes.items():
        setattr(env.processor, key, value)
    updates = []
    result = env.processor.run(
        env.task,
        save_task=env.saved.append,
        model_budget=2,
        cancel_event=changes.get("cancel"),
        user_turn_id="turn-batch",
        progress=updates.append,
    )
    return result, updates


def test_twelve_papers_auto_preview_without_manual_continue_or_analysis(tmp_path):
    env = setup(tmp_path, 12)
    result, updates = batch(env)
    assert len(env.saved[-1].previews) == 12
    assert env.llm.calls == result.model_call_count == 12
    assert result.final_status == "needs_user_input"
    assert not env.saved[-1].extraction_snapshots
    assert all(d.action == "hold" for d in env.saved[-1].preview_decisions.values())
    assert any("12/12" in text for text in updates)
    assert any("第2批" in text for text in updates)


def test_service_failure_does_not_auto_retry(tmp_path):
    env = setup(tmp_path, 12)
    env.llm.fail = True
    result, _ = batch(env)
    assert result.error["code"] == "FULLTEXT_PREVIEW_FAILED"
    assert env.llm.calls == 1
    assert env.saved[-1].stage == "failed"


def test_cancel_keeps_first_success_and_starts_no_second_request(tmp_path):
    env = setup(tmp_path, 12)
    cancel = threading.Event()
    env.llm.cancel = cancel
    result, _ = batch(env, cancel=cancel)
    assert result.status == "cancelled"
    assert env.llm.calls == 1
    assert len(env.saved[-1].previews) == 1


def test_batch_ceiling_preserves_partial_result(tmp_path):
    env = setup(tmp_path, 12)
    result, _ = batch(env, max_batches=2)
    assert env.llm.calls == 4
    assert len(env.saved[-1].previews) == 4
    assert result.error and "批次上限" in result.response_text


def test_confirmation_after_preview_can_select_detailed_analysis(tmp_path):
    env = setup(tmp_path, 2)
    batch(env)
    task = env.saved[-1].model_copy(
        update={
            "user_instructions": (*env.saved[-1].user_instructions, "确认分析这些论文")
        }
    )
    result = env.processor.run(
        task,
        save_task=env.saved.append,
        model_budget=6,
        cancel_event=None,
        user_turn_id="confirmed",
    )
    assert result.error["code"] == "FULLTEXT_EXTRACTION_UNAVAILABLE"
    assert all(d.action == "extract" for d in env.saved[-1].preview_decisions.values())
    assert env.llm.calls == 2


def test_original_analysis_request_cannot_skip_new_paper_confirmation(tmp_path):
    env = setup(tmp_path, 2)
    env.task = env.task.model_copy(
        update={"user_instructions": ("上传后提取指标并详细分析",)}
    )
    result, _ = batch(env)
    assert result.final_status == "needs_user_input"
    assert all(d.action == "hold" for d in env.saved[-1].preview_decisions.values())
    result = env.processor.run(
        env.saved[-1],
        save_task=env.saved.append,
        model_budget=6,
        cancel_event=None,
        user_turn_id="no-new-confirmation",
    )
    assert result.final_status == "needs_user_input"
    assert env.llm.calls == 2


def test_batch_budget_is_independent_of_exhausted_initial_graph_budget(tmp_path):
    env = setup(tmp_path, 2)
    env.processor.auto_batches = True
    env.processor.batch_model_budget = 6
    env.processor.require_preview_confirmation = True
    result = env.processor.run(
        env.task,
        save_task=env.saved.append,
        model_budget=0,
        cancel_event=None,
        user_turn_id="initial-graph-exhausted",
    )
    assert result.model_call_count == 2
    assert len(env.saved[-1].previews) == 2


def test_ambiguous_selection_saves_and_stops_without_progress_callback_failure(
    tmp_path,
):
    env = setup(tmp_path, 2)
    env.task = env.task.model_copy(update={"user_instructions": ("只预览第十篇",)})
    result, updates = batch(env)
    assert result.final_status == "needs_user_input"
    assert env.llm.calls == 0
    assert any("确认" in text for text in updates)


def test_stream_yields_before_operation_finishes_and_close_waits_for_safe_stop():
    from materials_screening.master.fulltext_batches import stream_saved_operation

    cancel = threading.Event()
    saved = threading.Event()

    def operation(progress):
        progress("已预览 1/12 篇")
        assert cancel.wait(timeout=5)
        saved.set()
        return "cancelled"

    stream = stream_saved_operation(operation, cancel)
    assert next(stream) == "已预览 1/12 篇"
    assert not saved.is_set()
    stream.close()
    assert cancel.is_set() and saved.is_set()


def test_stream_returns_result_and_propagates_failure_without_retry():
    from materials_screening.master.fulltext_batches import stream_saved_operation

    cancel = threading.Event()
    stream = stream_saved_operation(lambda progress: "done", cancel)
    with pytest.raises(StopIteration) as completed:
        next(stream)
    assert completed.value.value == "done"
    assert not cancel.is_set()

    def broken(progress):
        raise ValueError("offline test error")

    with pytest.raises(ValueError, match="offline test error"):
        next(stream_saved_operation(broken, cancel))


def test_fulltext_timeout_is_separate_and_validated(monkeypatch):
    from materials_screening.sub_agents.literature import fulltext_factory

    settings = []
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "45")
    monkeypatch.delenv("FULLTEXT_LLM_TIMEOUT_SECONDS", raising=False)
    monkeypatch.setattr(fulltext_factory, "create_llm_provider", settings.append)
    fulltext_factory._ConfiguredLlm()
    assert settings[0].llm_timeout_seconds == 120
    assert fulltext_factory.Settings().llm_timeout_seconds == 45
    monkeypatch.setenv("FULLTEXT_LLM_TIMEOUT_SECONDS", "0")
    with pytest.raises(ValueError, match="timeout"):
        fulltext_factory._ConfiguredLlm()


def test_analysis_budget_automatically_resumes_saved_two_stage_work(tmp_path):
    from tests.unit.master.test_fulltext_two_stage_continuation import environment

    env = environment(tmp_path)
    env.processor.auto_batches = True
    env.processor.batch_model_budget = 3
    env.processor.batch_seconds = 600
    result = env.processor.run(
        env.task,
        save_task=env.saved.append,
        model_budget=3,
        cancel_event=None,
        user_turn_id="analysis-batches",
    )
    assert result.status == "completed", result.model_dump()
    assert result.model_call_count == 6
    assert env.locate_calls == env.value_calls == 1
    assert env.saved[-1].stage == "finished"


def test_real_runner_sqlite_streams_twelve_saved_previews_in_one_user_turn(tmp_path):
    import sqlite3
    from unittest.mock import Mock

    from langgraph.checkpoint.sqlite import SqliteSaver

    from materials_screening.agent.conversation_store import SqliteConversationStore
    from materials_screening.master.ui_controller import stream_master_request
    from tests.unit.master.test_fulltext_task_state import make_runner, seed

    env = setup(tmp_path, 12)
    env.processor.auto_batches = True
    env.processor.batch_model_budget = 3
    env.processor.require_preview_confirmation = True
    store = SqliteConversationStore(tmp_path / "conversations.sqlite")
    model = Mock()
    try:
        with sqlite3.connect(
            tmp_path / "checkpoints.sqlite", check_same_thread=False
        ) as connection:
            runner = make_runner(
                tmp_path, store, SqliteSaver(connection), model, env.processor.artifacts
            )
            runner._preview_processor = env.processor
            seed(runner, store, task=env.task)
            updates = list(
                stream_master_request(
                    runner, message="只预览", conversation_id="conv-a"
                )
            )
            assert any(
                "12/12" in event.detail and not event.is_final for event in updates
            )
            assert updates[-1].result.final_status == "needs_user_input"
            assert updates[-1].result.model_call_count == 12
            saved = runner.get_fulltext_tasks("conv-a")[0]
            assert len(saved.previews) == 12
            assert saved.user_instructions == ("只预览",)
            assert store.get("conv-a").turn_count == 1
            model.generate.assert_not_called()
    finally:
        store.close()


def test_twelve_papers_extract_once_after_confirmation_and_preserve_snapshots(tmp_path):
    from tests.unit.master.test_fulltext_extraction import extraction_env

    env = extraction_env(tmp_path, 12)
    batch(env, batch_model_budget=3)
    assert env.matrix_llm.matrix_calls == 0
    task = env.saved[-1].model_copy(
        update={
            "user_instructions": (*env.saved[-1].user_instructions, "确认分析全部论文")
        }
    )
    result = env.processor.run(
        task,
        save_task=env.saved.append,
        model_budget=3,
        cancel_event=None,
        user_turn_id="confirmed-twelve",
    )
    assert result.error["code"] == "FULLTEXT_ANALYSIS_UNAVAILABLE"
    assert result.model_call_count == env.matrix_llm.matrix_calls == 12
    assert env.llm.calls == 12  # Preview results are not requested again.
    saved = env.saved[-1]
    assert len(saved.extraction_snapshots) == 12
    assert all(
        env.snapshots.load(ref).status == "complete"
        for ref in saved.extraction_snapshots.values()
    )
    result = env.processor.run(
        saved,
        save_task=env.saved.append,
        model_budget=3,
        cancel_event=None,
        user_turn_id="replay",
    )
    assert result.model_call_count == 0
    assert env.matrix_llm.matrix_calls == 12


def test_time_budget_auto_resumes_without_retrying_finished_previews(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from materials_screening.master import fulltext_preview

    env = setup(tmp_path, 3)
    now = [0]
    monkeypatch.setattr(
        fulltext_preview, "time", SimpleNamespace(monotonic=lambda: now[0])
    )
    generate = env.llm.generate_structured

    def slow(**kwargs):
        result = generate(**kwargs)
        now[0] += 700
        return result

    env.llm.generate_structured = slow
    result, updates = batch(env)
    assert result.final_status == "needs_user_input"
    assert env.llm.calls == result.model_call_count == 3
    assert len(env.saved[-1].previews) == 3
    assert any("第4批" in text for text in updates)


def test_no_progress_stops_without_unbounded_auto_loops(tmp_path):
    env = setup(tmp_path)
    env.processor.auto_batches = True
    result = env.processor.run(
        env.task,
        save_task=env.saved.append,
        model_budget=0,
        cancel_event=None,
        user_turn_id="zero-batch-budget",
    )
    assert "没有产生新进度" in result.response_text
    assert len(env.saved) == 4
    assert env.llm.calls == 0


def test_staged_steps_count_as_progress_without_legacy_matrix_snapshots():
    from types import SimpleNamespace

    from materials_screening.agent.models import AgentResult
    from materials_screening.master.fulltext_batches import run_saved_batches

    task = SimpleNamespace(
        previews=(),
        extraction_snapshots={},
        fulltext_analysis_ref=None,
        staged_extractions={"doc": object()},
        resume_stage="extracting",
        step=0,
    )
    batches = []

    def single(current, *, save_task, **kwargs):
        step = current.step + 1
        batches.append(step)
        save_task(SimpleNamespace(**{**vars(current), "step": step}))
        return AgentResult(
            conversation_id="unit",
            user_turn_id="unit",
            status="completed",
            final_status="completed",
            response_text="saved",
            error={"code": "FULLTEXT_EXTRACTION_BUDGET"} if step < 4 else None,
        )

    processor = SimpleNamespace(
        batch_model_budget=3,
        max_batches=10,
        _run_single_batch=single,
        extraction_processor=SimpleNamespace(
            progress_signature=lambda value: {"step": value.step}
        ),
    )
    result = run_saved_batches(
        processor,
        task,
        save_task=lambda _: None,
        model_budget=3,
        cancel_event=None,
        user_turn_id="unit",
    )
    assert not result.error and batches == [1, 2, 3, 4]
