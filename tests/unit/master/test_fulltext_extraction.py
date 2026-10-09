"""Snapshot isolation and real generic extraction with offline candidates."""

import hashlib
import threading

import pytest

from materials_screening.data_analysis.dataset_store import DatasetStore
from materials_screening.llm.base import StructuredProviderResponse
from materials_screening.master.fulltext_extraction import FulltextExtractionProcessor
from materials_screening.master.fulltext_snapshots import ExtractionSnapshotStore
from materials_screening.sub_agents.literature.matrix_automation import (
    GroupCandidate,
    MatrixExtractionBatch,
    MeasurementCandidate,
)
from tests.unit.master.test_fulltext_preview import run, setup


class MatrixLlm:
    def __init__(self, preview_llm, chunks):
        self.preview_llm, self.chunks = preview_llm, chunks
        self.matrix_calls = 0
        self.cancel = None

    def generate_structured(self, **kwargs):
        if kwargs["output_model"] is not MatrixExtractionBatch:
            return self.preview_llm.generate_structured(**kwargs)
        self.matrix_calls += 1
        chunk = next(row for row in self.chunks if row.chunk_id in kwargs["user_text"])
        batch = MatrixExtractionBatch(
            groups=(
                GroupCandidate(
                    group_key="sample-a",
                    label="Sample A",
                    material="sample",
                    role="treatment",
                    conditions={"temperature": "25 C"},
                    source_quote=chunk.text,
                    chunk_id=chunk.chunk_id,
                ),
            ),
            measurements=(
                MeasurementCandidate(
                    group_key="sample-a",
                    metric="strength",
                    value_text="12.0",
                    numeric_value=12.0,
                    unit="MPa",
                    source_quote=chunk.text,
                    chunk_id=chunk.chunk_id,
                ),
            ),
        )
        if self.cancel is not None:
            self.cancel.set()
        return StructuredProviderResponse(
            parsed=batch,
            provider="offline",
            model="unit",
            request_id="unit",
            latency_ms=0,
            input_tokens=1,
            output_tokens=1,
            reasoning_tokens=0,
            raw_output_sha256=None,
        )


def extraction_env(tmp_path, count=1, pages=1, body=None):
    import pymupdf

    from materials_screening.sub_agents.literature.rag import (
        PyMuPdfParser,
        parse_pdf_chunks,
    )

    env = setup(tmp_path, count)
    # Use a freshly registered actual PDF; never substitute text for checked source.
    refs, chunks = [], []
    for index in range(count):
        source = tmp_path / f"experiment-{index}.pdf"
        with pymupdf.open() as doc:
            for page_index in range(pages):
                doc.new_page().insert_text(
                    (72, 72),
                    body
                    if body is not None
                    else (
                        "Results: Sample A at 25 C had strength 12.0 MPa. "
                        f"Experiment {index}, page {page_index}."
                    ),
                )
            doc.save(source)
        artifact = env.processor.artifacts.register_pdf(
            source, conversation_id="conv-a"
        )
        refs.append(artifact.artifact_id)
        chunks.extend(
            parse_pdf_chunks(
                source,
                parser=PyMuPdfParser(),
                document_id="doc-" + artifact.metadata["sha256"][:24],
            )
        )
    env.task = env.task.model_copy(
        update={
            "artifact_refs": tuple(refs),
            "document_ids": tuple(
                "doc-" + ref.removeprefix("artifact-pdf-") for ref in refs
            ),
        }
    )
    env.store.chunks[:] = chunks
    env.llm.chunks = chunks
    env.matrix_llm = MatrixLlm(env.llm, chunks)
    env.processor.llm_factory = lambda: env.matrix_llm
    env.snapshots = ExtractionSnapshotStore(tmp_path / "snapshots")
    env.datasets = DatasetStore(tmp_path / "datasets")
    env.processor.extraction_processor = FulltextExtractionProcessor(
        artifacts=env.processor.artifacts,
        snapshots=env.snapshots,
        dataset_factory=lambda: env.datasets,
        model_profile="offline-unit",
    )
    return env


def test_normal_preview_continues_to_new_snapshot_and_trial_dataset(tmp_path):
    env = extraction_env(tmp_path)
    result = run(env)
    task = env.saved[-1]
    assert result.error["code"] == "FULLTEXT_ANALYSIS_UNAVAILABLE"
    assert task.resume_stage == "analyzing"
    ref = task.extraction_snapshots[task.document_ids[0]]
    snapshot = env.snapshots.load(ref)
    assert snapshot.status == "complete"
    assert len(snapshot.matrix.measurements) == 1
    assert snapshot.matrix.measurements[0].review_status == "pending"
    handoff = task.measurement_handoffs[task.document_ids[0]]
    frame = env.datasets.load_dataframe(handoff.dataset_id)
    assert len(frame) == 1 and frame.iloc[0]["numeric_value"] == 12.0
    assert frame.iloc[0]["analysis_mode"] == "trial"
    assert frame.iloc[0]["snapshot_id"] == ref.snapshot_id
    assert frame.iloc[0]["review_status"] == "pending"
    assert env.llm.calls == env.matrix_llm.matrix_calls == 1
    assert snapshot.pdf_sha256.startswith(task.document_ids[0].removeprefix("doc-"))


def test_successful_snapshot_restarts_without_new_model_calls_or_duplicate_files(
    tmp_path,
):
    env = extraction_env(tmp_path)
    run(env)
    first = env.saved[-1]
    files = set((tmp_path / "snapshots").glob("*.json"))
    env.processor.extraction_processor.snapshots = ExtractionSnapshotStore(
        tmp_path / "snapshots"
    )
    result = run(env, first)
    assert result.model_call_count == 0
    assert env.matrix_llm.matrix_calls == 1
    assert env.saved[-1].extraction_snapshots == first.extraction_snapshots
    assert env.saved[-1].measurement_handoffs == first.measurement_handoffs
    assert set((tmp_path / "snapshots").glob("*.json")) == files


def test_snapshot_bytes_cannot_be_overwritten_or_tampered_and_reused(tmp_path):
    env = extraction_env(tmp_path)
    run(env)
    ref = next(iter(env.saved[-1].extraction_snapshots.values()))
    snapshot = env.snapshots.load(ref)
    with pytest.raises(ValueError):
        env.snapshots.save(snapshot.model_copy(update={"title": "replacement"}))
    path = tmp_path / "snapshots" / f"{ref.snapshot_id}.json"
    old = hashlib.sha256(path.read_bytes()).hexdigest()
    assert old == ref.content_sha256
    path.write_text("{}", encoding="utf-8")  # Deliberately corrupted test fixture only.
    result = run(env, env.saved[-1])
    assert result.error["code"] == "FULLTEXT_EXTRACTION_FAILED"
    assert env.matrix_llm.matrix_calls == 1


def test_cancel_after_response_saves_raw_batch_but_does_not_register_dataset(tmp_path):
    env = extraction_env(tmp_path)
    cancelled = threading.Event()
    env.matrix_llm.cancel = cancelled
    result = run(env, cancel=cancelled)
    assert result.status == "cancelled"
    task = env.saved[-1]
    snapshot = env.snapshots.load(task.extraction_snapshots[task.document_ids[0]])
    assert len(snapshot.batches) == 1
    assert not task.measurement_handoffs
    assert task.resume_stage == "extracting"
    env.matrix_llm.cancel = None
    run(env, task)
    assert env.matrix_llm.matrix_calls == 1
    assert env.saved[-1].measurement_handoffs


def test_preview_budget_is_shared_with_extraction_and_resume_skips_preview(tmp_path):
    env = extraction_env(tmp_path)
    result = run(env, budget=1)
    assert result.error["code"] == "FULLTEXT_EXTRACTION_BUDGET"
    assert env.matrix_llm.matrix_calls == 0
    task = env.saved[-1]
    assert task.resume_stage == "extracting"
    run(env, task)
    assert env.llm.calls == 1 and env.matrix_llm.matrix_calls == 1


def test_new_request_policy_produces_new_immutable_attempt_not_old_snapshot(tmp_path):
    env = extraction_env(tmp_path)
    run(env)
    first = env.saved[-1]
    ref = first.extraction_snapshots[first.document_ids[0]]
    old = env.snapshots.load(ref)
    run(
        env, first.model_copy(update={"user_instructions": ("仅核对本论文报告的强度",)})
    )
    latest = env.snapshots.load(
        env.saved[-1].extraction_snapshots[first.document_ids[0]]
    )
    assert latest.snapshot_id != ref.snapshot_id
    assert latest.policy_sha256 != old.policy_sha256
    assert latest.parent_snapshot_id is not None
    assert env.snapshots.load(ref) == old
    assert env.matrix_llm.matrix_calls == 2


def test_only_preview_never_creates_snapshots_or_trial_data(tmp_path):
    env = extraction_env(tmp_path)
    run(env, env.task.model_copy(update={"user_instructions": ("只预览这些论文",)}))
    assert not env.saved[-1].extraction_snapshots
    assert env.matrix_llm.matrix_calls == 0
    assert not list((tmp_path / "snapshots").glob("*.json"))


def test_changing_selected_document_does_not_use_prior_other_handoffs(tmp_path):
    env = extraction_env(tmp_path, 3)
    run(env)
    first = env.saved[-1]
    assert len(first.measurement_handoffs) == 3
    result = run(env, first.model_copy(update={"user_instructions": ("只分析第2篇",)}))
    assert result.error["code"] == "FULLTEXT_ANALYSIS_UNAVAILABLE"
    assert set(env.saved[-1].preview_decisions) == {first.document_ids[1]}
    assert (
        env.saved[-1].measurement_handoffs[first.document_ids[0]]
        == first.measurement_handoffs[first.document_ids[0]]
    )


def test_multi_batch_budget_resume_reuses_successful_raw_batch(tmp_path):
    env = extraction_env(tmp_path, pages=3)
    env.processor.extraction_processor.batch_chars = 1
    result = run(env, budget=2)
    assert result.error["code"] == "FULLTEXT_EXTRACTION_BUDGET"
    first = env.saved[-1]
    snapshot = env.snapshots.load(first.extraction_snapshots[first.document_ids[0]])
    assert len(snapshot.batches) == 1 and snapshot.planned_batches == 3
    assert snapshot.status == "partial"
    result = run(env, first)
    assert result.error["code"] == "FULLTEXT_ANALYSIS_UNAVAILABLE"
    latest = env.snapshots.load(
        env.saved[-1].extraction_snapshots[first.document_ids[0]]
    )
    assert latest.status == "complete" and len(latest.batches) == 3
    assert env.llm.calls == 1 and env.matrix_llm.matrix_calls == 3
    assert (
        env.snapshots.load(first.extraction_snapshots[first.document_ids[0]])
        == snapshot
    )


def test_cached_handoff_revalidates_dataset_and_never_reregisters(
    tmp_path, monkeypatch
):
    env = extraction_env(tmp_path)
    run(env)
    first = env.saved[-1]
    monkeypatch.setattr(
        env.datasets,
        "register_records",
        lambda *args, **kwargs: pytest.fail("Repeated registration"),
    )
    result = run(env, first)
    assert result.error["code"] == "FULLTEXT_ANALYSIS_UNAVAILABLE"
    assert result.model_call_count == 0


def test_dataset_tampering_fails_without_new_model_request(tmp_path):
    env = extraction_env(tmp_path)
    run(env)
    first = env.saved[-1]
    handoff = next(iter(first.measurement_handoffs.values()))
    path = env.datasets.resolve_path(handoff.dataset_id)
    path.write_text("[]", encoding="utf-8")  # Corrupt test output only.
    result = run(env, first)
    assert result.error["code"] == "FULLTEXT_EXTRACTION_FAILED"
    assert result.model_call_count == 0


def test_schema_failed_batch_is_partial_and_only_missing_batch_retries(tmp_path):
    from materials_screening.llm.errors import LLMStructuredOutputError

    env = extraction_env(tmp_path, pages=2)
    env.processor.extraction_processor.batch_chars = 1
    original = env.matrix_llm.generate_structured
    failed_requests = []
    first_chunk = env.store.chunks[0].chunk_id

    def failing(**kwargs):
        if (
            kwargs["output_model"] is MatrixExtractionBatch
            and first_chunk in kwargs["user_text"]
        ):
            failed_requests.append(1)
            raise LLMStructuredOutputError("offline schema failure")
        return original(**kwargs)

    env.matrix_llm.generate_structured = failing
    result = run(env)
    assert result.error["code"] == "FULLTEXT_EXTRACTION_FAILED"
    first = env.saved[-1]
    snapshot = env.snapshots.load(first.extraction_snapshots[first.document_ids[0]])
    assert snapshot.status == "partial" and len(snapshot.batches) == 1
    assert not first.measurement_handoffs and len(failed_requests) == 2
    assert env.matrix_llm.matrix_calls == 1
    env.matrix_llm.generate_structured = original
    result = run(env, first)
    assert result.error["code"] == "FULLTEXT_ANALYSIS_UNAVAILABLE"
    assert result.model_call_count == 1 and env.matrix_llm.matrix_calls == 2


def test_source_replaced_after_response_never_hands_off(tmp_path):
    env = extraction_env(tmp_path)
    original = env.matrix_llm.generate_structured

    def replacing(**kwargs):
        response = original(**kwargs)
        if kwargs["output_model"] is MatrixExtractionBatch:
            artifact = env.processor.artifacts.get(env.task.artifact_refs[0])
            env.processor.artifacts.resolve_path(artifact.artifact_id).write_bytes(
                b"%PDF-replaced-test"
            )
        return response

    env.matrix_llm.generate_structured = replacing
    result = run(env)
    assert result.error["code"] == "FULLTEXT_EXTRACTION_FAILED"
    assert not env.saved[-1].measurement_handoffs
    assert env.saved[-1].resume_stage == "extracting"


@pytest.mark.parametrize(
    "value_text,numeric", [("12.0-14.0", 13.0), (">12.0", 12.0), ("12.0", 999.0)]
)
def test_ranges_bounds_and_inconsistent_values_never_become_trial_scalars(
    tmp_path, value_text, numeric
):
    from uuid import uuid4

    from materials_screening.master.fulltext_handoff import snapshot_trial_handoff

    env = extraction_env(tmp_path)
    run(env)
    snapshot = env.snapshots.load(
        next(iter(env.saved[-1].extraction_snapshots.values()))
    )
    measurement = snapshot.matrix.measurements[0].model_copy(
        update={"value_text": value_text, "numeric_value": numeric}
    )
    candidate = snapshot.model_copy(
        update={
            "snapshot_id": "snapshot-" + uuid4().hex,
            "matrix": snapshot.matrix.model_copy(
                update={"measurements": (measurement,)}
            ),
        }
    )
    ref = env.snapshots.save(candidate)
    loaded = env.snapshots.load(ref)
    handoff = snapshot_trial_handoff(
        loaded,
        chunks=env.store.chunks,
        dataset_factory=lambda: pytest.fail("Empty unsafe dataset must not register"),
    )
    assert handoff.record_count == 0 and handoff.dataset_id is None
    assert handoff.isolated_measurement_ids == (measurement.measurement_id,)
    assert handoff.isolation_reasons[measurement.measurement_id]
    assert loaded.matrix.measurements[0].value_text == value_text


def test_readable_extraction_displays_rows_sources_and_pending_boundary(tmp_path):
    env = extraction_env(tmp_path)
    result = run(env)
    assert "Sample A" in result.response_text
    assert "12.0 MPa" in result.response_text
    assert "pending" in result.response_text
    assert "原文" in result.response_text and "第 1 页" in result.response_text
    assert "25 C" in result.response_text


def test_matching_unit_suffix_does_not_bypass_sample_binding(tmp_path):
    from materials_screening.master.fulltext_handoff import snapshot_trial_handoff

    env = extraction_env(tmp_path)
    run(env)
    snapshot = env.snapshots.load(
        next(iter(env.saved[-1].extraction_snapshots.values()))
    )
    row = snapshot.matrix.measurements[0]
    measurement = row.model_copy(update={"value_text": "12.0 MPa", "unit": "MPa"})
    candidate = snapshot.model_copy(
        update={
            "matrix": snapshot.matrix.model_copy(
                update={"measurements": (measurement,)}
            )
        }
    )
    before = candidate.model_dump(mode="json")
    handoff = snapshot_trial_handoff(
        candidate, chunks=env.store.chunks, dataset_factory=lambda: env.datasets
    )
    assert handoff.record_count == 0
    assert handoff.isolation_reasons[measurement.measurement_id] == (
        "sample_value_binding_rejected",
    )
    assert candidate.model_dump(mode="json") == before


def test_snapshot_cannot_promote_pending_rows_or_import_foreign_document(tmp_path):
    env = extraction_env(tmp_path)
    run(env)
    snapshot = env.snapshots.load(
        next(iter(env.saved[-1].extraction_snapshots.values()))
    )
    measurement = snapshot.matrix.measurements[0]
    for update in ({"review_status": "approved"}, {"document_id": "doc-" + "a" * 24}):
        invalid = snapshot.model_copy(
            update={
                "matrix": snapshot.matrix.model_copy(
                    update={"measurements": (measurement.model_copy(update=update),)}
                )
            }
        )
        with pytest.raises(ValueError):
            env.snapshots.save(invalid)


def test_input_batch_size_tracks_output_budget_without_raising_limits(tmp_path):
    env = extraction_env(tmp_path)
    processor = FulltextExtractionProcessor(
        artifacts=env.processor.artifacts,
        snapshots=env.snapshots,
        dataset_factory=lambda: env.datasets,
        model_profile="offline-unit",
        max_output_tokens=4096,
        batch_chars=8000,
    )
    assert processor.batch_chars == processor.max_output_tokens == 4096
    assert processor.max_evidence_chars == 24000


@pytest.mark.parametrize("stream", [False, True])
def test_real_master_checkpoint_reopens_snapshot_and_dataset_without_model(
    tmp_path, stream
):
    import sqlite3
    from unittest.mock import Mock

    from langgraph.checkpoint.sqlite import SqliteSaver

    from materials_screening.agent.conversation_store import SqliteConversationStore
    from materials_screening.master.artifact_registry import ArtifactRegistry
    from materials_screening.master.master_runner import MasterAgentRunner
    from materials_screening.master.master_settings import MasterAgentSettings
    from materials_screening.master.sub_agent_registry import SubAgentRegistry
    from tests.unit.master.test_fulltext_task_state import seed

    env = extraction_env(tmp_path)
    model = Mock()

    def make(store, connection):
        return MasterAgentRunner(
            settings=MasterAgentSettings(),
            store=store,
            sub_agent_registry=SubAgentRegistry(),
            master_model=model,
            checkpointer=SqliteSaver(connection),
            artifact_registry=env.processor.artifacts,
            preview_processor=env.processor,
        )

    def ask(runner):
        kwargs = {
            "message": "继续",
            "conversation_id": "conv-a",
            "task_id": env.task.task_id,
        }
        return (
            list(runner.ask_stream(**kwargs))[-1].result
            if stream
            else runner.ask(**kwargs)
        )

    store = SqliteConversationStore(tmp_path / "conversations.sqlite")
    with sqlite3.connect(
        tmp_path / "checkpoints.sqlite", check_same_thread=False
    ) as connection:
        runner = make(store, connection)
        seed(runner, store, task=env.task)
        result = ask(runner)
        assert result.error["code"] == "FULLTEXT_ANALYSIS_UNAVAILABLE"
        first = runner.get_fulltext_tasks("conv-a")[0]
        assert first.extraction_snapshots and first.measurement_handoffs
    store.close()
    env.processor.artifacts = ArtifactRegistry(tmp_path / "artifacts")
    env.processor.extraction_processor.artifacts = env.processor.artifacts
    env.processor.extraction_processor.snapshots = ExtractionSnapshotStore(
        tmp_path / "snapshots"
    )
    env.datasets = DatasetStore(tmp_path / "datasets")
    store = SqliteConversationStore(tmp_path / "conversations.sqlite")
    with sqlite3.connect(
        tmp_path / "checkpoints.sqlite", check_same_thread=False
    ) as connection:
        runner = make(store, connection)
        assert runner.get_fulltext_tasks("conv-a") == (first,)
        result = ask(runner)
        assert result.model_call_count == 0
        assert result.error["code"] == "FULLTEXT_ANALYSIS_UNAVAILABLE"
        assert (
            runner.get_fulltext_tasks("conv-a")[0].extraction_snapshots
            == first.extraction_snapshots
        )
        assert store.get("conv-a").turn_count == 2
    store.close()
    model.generate.assert_not_called()
    assert env.llm.calls == env.matrix_llm.matrix_calls == 1
