"""Real PDF parsing + preview service with offline LLM; not scientific validation."""

import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest

from materials_screening.llm.base import StructuredProviderResponse
from materials_screening.llm.errors import LLMConnectionError, LLMStructuredOutputError
from materials_screening.master.artifact_registry import ArtifactRegistry
from materials_screening.master.fulltext_preview import FulltextPreviewProcessor
from materials_screening.master.fulltext_tasks import new_fulltext_task
from materials_screening.sub_agents.literature.preview import PreviewCandidate
from materials_screening.sub_agents.literature.rag import (
    PyMuPdfParser,
    parse_pdf_chunks,
)

QUESTION = "查询材料并分析稳定性；检索实验论文，上传全文后提取实验指标并统计。"


class OfflineLlm:
    def __init__(self, chunks):
        self.chunks = chunks
        self.calls = 0
        self.fail = False
        self.cancel = None

    def generate_structured(self, **kwargs):
        self.calls += 1
        if self.fail:
            raise LLMConnectionError("DO NOT EXPOSE private provider details")
        chunk = next(
            chunk for chunk in self.chunks if chunk.chunk_id in kwargs["user_text"]
        )
        if self.cancel is not None:
            self.cancel.set()
        return StructuredProviderResponse(
            parsed=PreviewCandidate(
                article_type="research",
                topic_relevance="core",
                research_question="研究材料样品的实验表征。",
                methods="采用实验表征方法。",
                key_findings="作者报告了样品实验表征结果。",
                recommendation="deep_analyze",
                reason="与本任务的实验表征相关。",
                evidence_quote=chunk.text,
                chunk_id=chunk.chunk_id,
            ),
            provider="offline-unit",
            model="unit",
            request_id="unit",
            latency_ms=0,
            input_tokens=1,
            output_tokens=1,
            reasoning_tokens=0,
            raw_output_sha256=None,
        )


class IndexedStore:
    def __init__(self, chunks):
        self.chunks = chunks

    def document_exists(self, document_id):
        return any(chunk.document_id == document_id for chunk in self.chunks)

    def get_document_chunks(self, document_id):
        return [chunk for chunk in self.chunks if chunk.document_id == document_id]


def setup(tmp_path, count=1):
    pymupdf = pytest.importorskip("pymupdf")
    artifacts = ArtifactRegistry(tmp_path / "artifacts")
    refs, chunks = [], []
    for index in range(count):
        source = tmp_path / f"paper-{index}.pdf"
        document = pymupdf.open()
        page = document.new_page()
        page.insert_text(
            (72, 72), f"Material sample {index}: experimental characterization results."
        )
        document.save(source)
        document.close()
        ref = artifacts.register_pdf(source, conversation_id="conv-a")
        refs.append(ref.artifact_id)
        doc_id = "doc-" + ref.metadata["sha256"][:24]
        chunks.extend(
            parse_pdf_chunks(source, parser=PyMuPdfParser(), document_id=doc_id)
        )
    task = new_fulltext_task("conv-a", QUESTION).model_copy(
        update={
            "artifact_refs": tuple(refs),
            "document_ids": tuple(
                "doc-" + ref.removeprefix("artifact-pdf-") for ref in refs
            ),
        }
    )
    store, llm = IndexedStore(chunks), OfflineLlm(chunks)
    saved = []

    def must_not_index(_store):
        pytest.fail(
            "Already-indexed sources must not load embeddings or overwrite chunks"
        )

    processor = FulltextPreviewProcessor(
        artifacts=artifacts,
        store_factory=lambda: store,
        rag_factory=must_not_index,
        llm_factory=lambda: llm,
    )
    return SimpleNamespace(
        task=task, store=store, llm=llm, saved=saved, processor=processor
    )


def run(env, task=None, budget=6, cancel=None):
    return env.processor.run(
        task or env.task,
        save_task=env.saved.append,
        model_budget=budget,
        cancel_event=cancel,
        user_turn_id="turn-a",
    )


def retry(env, task, operation="retry-1", expected=None):
    return env.processor.regenerate(
        task,
        document_id=task.document_ids[0],
        operation_id=operation,
        expected=expected or env.processor.retry_token(task, task.document_ids[0]),
        save_task=env.saved.append,
        model_budget=3,
        cancel_event=None,
    )


def failed_preview(env):
    run(env, env.task.model_copy(update={"user_instructions": ("只预览",)}))
    task = env.saved[-1]
    preview = task.previews[0].model_copy(update={"evidence_quality": "fallback_chunk"})
    return task.model_copy(update={"previews": (preview,)})


def test_manual_preview_retry_preserves_history_and_requires_new_choice(tmp_path):
    env = setup(tmp_path)
    old = failed_preview(env)
    updated = retry(env, old)
    assert env.llm.calls == 2
    assert updated.preview_retry_history[0].previous_preview == old.previews[0]
    assert updated.preview_retry_history[0].status == "succeeded"
    assert updated.preview_decisions[old.document_ids[0]].action == "hold"
    assert not updated.extraction_snapshots
    run(env, updated.model_copy(update={"user_instructions": ("详细分析",)}))
    assert env.saved[-1].preview_decisions[old.document_ids[0]].action == "hold"


def test_retry_duplicate_restart_limit_and_private_error(tmp_path):
    env = setup(tmp_path)
    old = failed_preview(env)
    token = env.processor.retry_token(old, old.document_ids[0])
    env.llm.fail = True
    first = retry(env, old, expected=token)
    assert first.preview_retry_history[0].status == "failed"
    assert "private" not in first.model_dump_json()
    assert first.previews == old.previews
    calls = env.llm.calls
    assert retry(env, first, expected=token) == first
    assert env.llm.calls == calls
    recovered = type(old).model_validate_json(first.model_dump_json())
    second = retry(env, recovered, operation="retry-2")
    with pytest.raises(ValueError, match="2"):
        retry(env, second, operation="retry-3")
    assert env.llm.calls == calls + 1


@pytest.mark.parametrize(
    "stage,resume",
    [("extracting", None), ("finished", None), ("ready_to_resume", "extracting")],
)
def test_retry_cannot_reset_extraction_or_completed_tasks(tmp_path, stage, resume):
    env = setup(tmp_path)
    task = failed_preview(env).model_copy(
        update={"stage": stage, "resume_stage": resume}
    )
    with pytest.raises(ValueError):
        retry(env, task)
    assert env.llm.calls == 1


def test_new_explicit_choice_after_retry_can_extract(tmp_path):
    env = setup(tmp_path)
    updated = retry(env, failed_preview(env))
    updated = updated.model_copy(
        update={"user_instructions": (*updated.user_instructions, "详细分析第1篇")}
    )
    run(env, updated)
    assert env.saved[-1].preview_decisions[updated.document_ids[0]].action == "extract"
    assert env.llm.calls == 2


def test_started_retry_survives_interruption_without_reissuing_model(tmp_path):
    env = setup(tmp_path)
    old = failed_preview(env)
    token = env.processor.retry_token(old, old.document_ids[0])

    def interrupted_factory():
        raise KeyboardInterrupt()

    env.processor.llm_factory = interrupted_factory
    with pytest.raises(KeyboardInterrupt):
        retry(env, old, expected=token)
    interrupted = type(old).model_validate_json(env.saved[-1].model_dump_json())
    assert interrupted.preview_retry_history[-1].status == "started"
    env.processor.llm_factory = lambda: pytest.fail(
        "must not replay interrupted remote request"
    )
    assert retry(env, interrupted, expected=token) == interrupted


def test_retry_rejects_stale_scope_and_source_mutation(tmp_path):
    env = setup(tmp_path)
    old = failed_preview(env)
    with pytest.raises(ValueError, match="状态"):
        retry(env, old, expected="a" * 64)
    assert env.llm.calls == 1
    env.store.chunks = [replace(env.store.chunks[0], text="tampered source")]
    failed = retry(env, old)
    assert failed.preview_retry_history[-1].status == "failed"
    assert env.llm.calls == 1 and failed.previews == old.previews


def test_auto_preview_then_extract_checkpoint_not_fake_completion(tmp_path):
    env = setup(tmp_path)
    result = run(env)
    task = env.saved[-1]
    assert env.llm.calls == 1
    assert len(task.previews) == 1
    assert task.original_question == QUESTION
    assert task.preview_decisions[task.document_ids[0]].action == "extract"
    assert task.stage == "ready_to_resume" and task.resume_stage == "extracting"
    assert result.error["code"] == "FULLTEXT_EXTRACTION_UNAVAILABLE"
    assert "预览" in result.response_text and "未执行" in result.response_text


def test_resume_uses_persisted_preview_without_extra_model_calls(tmp_path):
    env = setup(tmp_path)
    run(env)
    processor = FulltextPreviewProcessor(
        artifacts=ArtifactRegistry(tmp_path / "artifacts"),
        store_factory=lambda: env.store,
        rag_factory=lambda _: pytest.fail("index rerun"),
        llm_factory=lambda: env.llm,
    )
    result = processor.run(
        env.saved[-1],
        save_task=env.saved.append,
        model_budget=6,
        cancel_event=None,
        user_turn_id="resume",
    )
    assert env.llm.calls == 1
    assert result.model_call_count == 0
    assert "复用" in result.response_text


def test_only_preview_saved_before_upload_blocks_auto_extraction(tmp_path):
    env = setup(tmp_path)
    task = env.task.model_copy(update={"user_instructions": ("只预览这些论文",)})
    result = run(env, task)
    saved = env.saved[-1]
    assert saved.stage == "awaiting_clarification"
    assert saved.preview_decisions[saved.document_ids[0]].action == "hold"
    assert result.final_status == "needs_user_input"
    assert "只预览" in result.response_text


def test_ordinal_selection_limits_which_documents_are_processed(tmp_path):
    env = setup(tmp_path, 3)
    task = env.task.model_copy(update={"user_instructions": ("只分析第2篇",)})
    run(env, task)
    saved = env.saved[-1]
    assert len(saved.previews) == 1
    assert saved.previews[0].document_id == saved.document_ids[1]
    assert env.llm.calls == 1


def test_invalid_ordinal_clarifies_without_model_or_index_calls(tmp_path):
    env = setup(tmp_path)
    result = run(
        env, env.task.model_copy(update={"user_instructions": ("只分析第2篇",)})
    )
    assert result.final_status == "needs_user_input"
    assert env.llm.calls == 0


def test_budget_stops_and_preserves_each_successful_preview(tmp_path):
    env = setup(tmp_path, 3)
    result = run(env, budget=1)
    assert result.error["code"] == "FULLTEXT_PREVIEW_BUDGET"
    saved = env.saved[-1]
    assert len(saved.previews) == 1
    assert saved.stage == "ready_to_resume" and saved.resume_stage == "previewing"
    run(env, saved)
    assert len(env.saved[-1].previews) == 3
    assert env.llm.calls == 3


def test_cancel_after_first_response_preserves_it_but_starts_no_next_call(tmp_path):
    env = setup(tmp_path, 3)
    cancelled = threading.Event()
    env.llm.cancel = cancelled
    result = run(env, cancel=cancelled)
    assert result.status == "cancelled"
    assert env.llm.calls == 1
    assert len(env.saved[-1].previews) == 1
    assert env.saved[-1].resume_stage == "previewing"


def test_connection_failure_not_converted_to_scope_clarification(tmp_path):
    env = setup(tmp_path)
    env.llm.fail = True
    result = run(env)
    assert result.error["code"] == "FULLTEXT_PREVIEW_FAILED"
    assert env.saved[-1].stage == "failed"
    assert env.llm.calls == 1
    assert "private" not in str(result.model_dump())
    assert result.final_status == "error"
    assert "请改写" not in result.response_text


def test_changed_index_source_rejected_before_model_without_overwriting(tmp_path):
    env = setup(tmp_path)
    original = env.store.chunks[0]
    env.store.chunks[0] = replace(original, text="injected foreign text")
    result = run(env)
    assert result.error["code"] == "FULLTEXT_PREVIEW_FAILED"
    assert env.llm.calls == 0
    assert env.store.chunks[0].text == "injected foreign text"


def test_normal_master_stream_runs_adapter_and_survives_sqlite_restart(tmp_path):
    import sqlite3
    from unittest.mock import Mock

    from langgraph.checkpoint.sqlite import SqliteSaver

    from materials_screening.agent.conversation_store import SqliteConversationStore
    from materials_screening.master.master_runner import MasterAgentRunner
    from materials_screening.master.master_settings import MasterAgentSettings
    from materials_screening.master.sub_agent_registry import SubAgentRegistry
    from tests.unit.master.test_fulltext_task_state import seed

    env = setup(tmp_path)
    model = Mock()
    for iteration in range(2):
        store = SqliteConversationStore(tmp_path / "conversations.sqlite")
        with sqlite3.connect(
            tmp_path / "checkpoints.sqlite", check_same_thread=False
        ) as connection:
            runner = MasterAgentRunner(
                settings=MasterAgentSettings(),
                store=store,
                sub_agent_registry=SubAgentRegistry(),
                master_model=model,
                checkpointer=SqliteSaver(connection),
                artifact_registry=env.processor.artifacts,
                preview_processor=env.processor,
            )
            if iteration == 0:
                seed(runner, store, task=new_fulltext_task("conv-a", QUESTION))
            events = list(
                runner.ask_stream(
                    message="继续",
                    conversation_id="conv-a",
                    artifact_refs=env.task.artifact_refs if iteration == 0 else (),
                )
            )
            assert any(
                not event.is_final and event.node == "fulltext_preview"
                for event in events
            )
            assert events[-1].result.error["code"] == "FULLTEXT_EXTRACTION_UNAVAILABLE"
            task = runner.get_fulltext_tasks("conv-a")[0]
            assert len(task.previews) == 1
            assert task.resume_stage == "extracting"
        store.close()
    assert env.llm.calls == 1
    model.generate.assert_not_called()


@pytest.mark.parametrize(
    "instruction",
    [
        "不要分析第2篇",
        "第2篇不要分析",
        "不分析第二篇",
        "第二篇暂时不要分析",
    ],
)
def test_negative_ordinal_does_not_select_excluded_paper(tmp_path, instruction):
    env = setup(tmp_path, 3)
    run(env, env.task.model_copy(update={"user_instructions": (instruction,)}))
    assert {row.document_id for row in env.saved[-1].previews} == {
        env.task.document_ids[0],
        env.task.document_ids[2],
    }


def test_ambiguous_negated_only_selection_stops_before_processing(tmp_path):
    env = setup(tmp_path, 3)
    result = run(
        env, env.task.model_copy(update={"user_instructions": ("不要只分析第2篇",)})
    )
    assert result.final_status == "needs_user_input"
    assert env.llm.calls == 0


def test_latest_detail_instruction_releases_preview_hold_without_model_rerun(tmp_path):
    env = setup(tmp_path)
    run(env, env.task.model_copy(update={"user_instructions": ("只预览这些论文",)}))
    task = env.saved[-1].model_copy(
        update={
            "user_instructions": (
                "只预览这些论文",
                "继续详细分析",
            )
        }
    )
    result = run(env, task)
    assert result.error["code"] == "FULLTEXT_EXTRACTION_UNAVAILABLE"
    assert env.saved[-1].preview_decisions[env.task.document_ids[0]].action == "extract"
    assert env.llm.calls == 1


def test_preview_response_respects_regular_output_cap(tmp_path):
    env = setup(tmp_path, 3)
    env.processor.max_output_bytes = 1024
    result = run(env)
    assert len(result.response_text.encode()) <= 1024
    assert result.warnings
    assert len(env.saved[-1].previews) == 3


@pytest.mark.parametrize("instruction", ["只分析第一、第三篇", "只分析第1、第3篇"])
def test_chinese_and_repeated_ordinals_are_supported(tmp_path, instruction):
    env = setup(tmp_path, 3)
    run(env, env.task.model_copy(update={"user_instructions": (instruction,)}))
    assert {row.document_id for row in env.saved[-1].previews} == {
        env.task.document_ids[0],
        env.task.document_ids[2],
    }


def test_exhausted_schema_repair_is_service_failure_not_scope_question(tmp_path):
    env = setup(tmp_path)

    def invalid(**kwargs):
        env.llm.calls += 1
        raise LLMStructuredOutputError("private invalid response")

    env.llm.generate_structured = invalid
    result = run(env)
    assert env.llm.calls == 2
    assert result.final_status == "error"
    assert result.error["code"] == "FULLTEXT_PREVIEW_FAILED"
    assert not env.saved[-1].preview_decisions
    assert "private" not in result.response_text


@pytest.mark.parametrize(
    "update", [{"evidence_quote": "foreign quote"}, {"preview_version": "obsolete"}]
)
def test_cached_preview_is_rechecked_without_silent_regeneration(tmp_path, update):
    env = setup(tmp_path)
    run(env)
    task = env.saved[-1]
    task = task.model_copy(
        update={"previews": (task.previews[0].model_copy(update=update),)}
    )
    result = run(env, task)
    assert result.error["code"] == "FULLTEXT_PREVIEW_FAILED"
    assert env.llm.calls == 1
    assert "foreign quote" not in result.response_text


def test_zero_budget_does_not_parse_index_or_call_model(tmp_path):
    env = setup(tmp_path)
    env.processor.store_factory = lambda: pytest.fail("store loaded without budget")
    result = run(env, budget=0)
    assert result.error["code"] == "FULLTEXT_PREVIEW_BUDGET"
    assert env.llm.calls == 0


def test_missing_index_is_lazily_ingested_then_verified(tmp_path):
    env = setup(tmp_path)
    expected = list(env.store.chunks)
    env.store.chunks.clear()
    ingestions = []

    def ingest(paths, paper_id):
        ingestions.append(paths)
        env.store.chunks.extend(expected)

    env.processor.rag_factory = lambda _: SimpleNamespace(ingest=ingest)
    run(env)
    assert len(ingestions) == 1
    assert env.llm.calls == 1
    run(env, env.saved[-1])
    assert len(ingestions) == 1


def test_factory_is_lazy_and_mock_mode_never_enables_remote(tmp_path, monkeypatch):
    from materials_screening.sub_agents.literature import fulltext_factory

    for name in ("BgeM3EmbeddingProvider", "PgVectorLiteratureStore", "_ConfiguredLlm"):
        monkeypatch.setattr(
            fulltext_factory,
            name,
            lambda *args, **kwargs: pytest.fail("eager dependency"),
        )
    artifacts = ArtifactRegistry(tmp_path / "artifacts")
    assert (
        fulltext_factory.create_fulltext_preview_processor(
            artifacts, enable_remote=False
        )
        is None
    )
    assert (
        fulltext_factory.create_fulltext_preview_processor(
            artifacts, enable_remote=True
        )
        is not None
    )


def test_normal_factory_checks_root_even_for_an_indexed_pdf(tmp_path, monkeypatch):
    from materials_screening.sub_agents.literature.fulltext_factory import (
        _ConfiguredParser,
    )

    env = setup(tmp_path)
    monkeypatch.setenv("LITERATURE_INGEST_ROOTS", str(tmp_path / "other-root"))
    env.processor.parser = _ConfiguredParser()
    result = run(env)
    assert result.error["code"] == "FULLTEXT_PREVIEW_FAILED"
    assert env.llm.calls == 0


@pytest.mark.parametrize("stream", [False, True])
def test_first_request_uses_remaining_budget_and_preserves_graph_answer(
    tmp_path, stream
):
    import json
    import sqlite3
    from unittest.mock import Mock

    from langgraph.checkpoint.sqlite import SqliteSaver

    from materials_screening.agent.conversation_store import SqliteConversationStore
    from materials_screening.agent.model_base import (
        AgentModelStatus,
        MaterialAgentResponse,
    )
    from materials_screening.agent.models import AgentMessageItem
    from materials_screening.master.master_runner import MasterAgentRunner
    from materials_screening.master.master_settings import MasterAgentSettings
    from materials_screening.master.sub_agent_registry import SubAgentRegistry
    from tests.unit.master.test_fulltext_task_state import QUESTION as INITIAL_QUESTION

    env = setup(tmp_path)
    env.processor.store_factory = lambda: pytest.fail("no remaining model budget")
    model = Mock()
    model.generate.return_value = MaterialAgentResponse(
        status=AgentModelStatus.COMPLETED,
        provider="unit",
        model="offline-unit",
        request_id="unit",
        output_items=(
            AgentMessageItem(
                role="assistant",
                content=json.dumps(
                    {
                        "status": "needs_user_input",
                        "answer": "第一阶段测试回答",
                        "referenced_material_ids": [],
                        "evidence_ids": [],
                        "warnings": [],
                        "follow_up_question": "请提供PDF",
                    },
                    ensure_ascii=False,
                ),
            ),
        ),
    )
    store = SqliteConversationStore(tmp_path / "conversations.sqlite")
    store.create("conv-a")
    with sqlite3.connect(
        tmp_path / "checkpoints.sqlite", check_same_thread=False
    ) as connection:
        runner = MasterAgentRunner(
            settings=MasterAgentSettings(master_max_model_calls_per_turn=1),
            store=store,
            sub_agent_registry=SubAgentRegistry(),
            master_model=model,
            checkpointer=SqliteSaver(connection),
            artifact_registry=env.processor.artifacts,
            preview_processor=env.processor,
        )
        args = {
            "message": INITIAL_QUESTION,
            "conversation_id": "conv-a",
            "artifact_refs": env.task.artifact_refs,
        }
        result = (
            list(runner.ask_stream(**args))[-1].result if stream else runner.ask(**args)
        )
        assert result.error["code"] == "FULLTEXT_PREVIEW_BUDGET"
        assert "第一阶段测试回答" in result.response_text
        assert result.model_call_count == 1
        assert env.llm.calls == 0
        assert store.get("conv-a").turn_count == 1
        task = runner.get_fulltext_tasks("conv-a")[0]
        assert task.original_question == INITIAL_QUESTION
        assert task.resume_stage == "previewing"
    store.close()
