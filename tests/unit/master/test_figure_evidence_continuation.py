"""Real checked PDF, snapshots and statistics graph; LLM proposals are offline."""

import hashlib

from materials_screening.master.figure_evidence_contracts import PageRegion
from materials_screening.master.figure_evidence_review import FigureEvidenceService
from materials_screening.master.figure_evidence_store import FigureEvidenceStore
from tests.unit.master.test_fulltext_analysis import analysis_env
from tests.unit.master.test_fulltext_preview import run


def test_figure_gate_then_actual_statistics_with_wrong_text_keeps_inputs(tmp_path):
    def configure(env):
        env.task = env.task.model_copy(
            update={"figure_review_policy": "figure-evidence-review-v1"}
        )
        env.figure_service = FigureEvidenceService(
            FigureEvidenceStore(tmp_path / "figures"),
            env.processor.artifacts,
            env.snapshots,
        )
        env.processor.extraction_processor.figure_review_service = env.figure_service

    env = analysis_env(tmp_path, configure=configure)
    analysis = env.processor.extraction_processor.analysis_processor
    analysis.figure_review_service = env.figure_service
    task = env.task
    assert task.stage == "awaiting_figure_review" and task.fulltext_analysis_ref is None
    before = env.datasets.load_dataframe(
        next(iter(task.measurement_handoffs.values())).dataset_id
    ).to_json(orient="records")
    repeated = run(env, task)
    assert (
        repeated.final_status == "needs_user_input" and repeated.model_call_count == 0
    )
    assert env.scope_calls == 0
    service = env.figure_service
    ref = service.add(
        task,
        conversation_id=task.conversation_id,
        expected=task.figure_evidence_ref,
        operation_id="add",
        document_id=task.document_ids[0],
        page=1,
        region=PageRegion(x0=20, y0=50, x1=200, y1=120),
        figure_label="unit manually selected",
        kind="axis_label",
    )
    task = task.model_copy(update={"figure_evidence_ref": ref})
    candidate = service.store.load(ref).candidates[0]
    ref = service.decide(
        task,
        conversation_id=task.conversation_id,
        expected=ref,
        operation_id="confirm",
        candidate_id=candidate.candidate_id,
        action="text_verified",
        text="INTENTIONALLY WRONG: 999 min",
    )
    task = task.model_copy(update={"figure_evidence_ref": ref})
    ref = service.close(
        task, conversation_id=task.conversation_id, expected=ref, operation_id="close"
    )
    task = task.model_copy(update={"figure_evidence_ref": ref})
    result = run(env, task)
    assert result.status == "completed" and result.tool_call_count == 2
    finished = env.saved[-1]
    assert finished.stage == "finished"
    record = env.analysis_store.load(finished.fulltext_analysis_ref)
    assert record.figure_evidence_ref == ref and record.analysis_ids
    assert "实验条件尚未绑定" in result.response_text
    assert "INTENTIONALLY WRONG" in result.response_text
    after = env.datasets.load_dataframe(
        next(iter(task.measurement_handoffs.values())).dataset_id
    ).to_json(orient="records")
    assert (
        hashlib.sha256(before.encode()).digest()
        == hashlib.sha256(after.encode()).digest()
    )
    scoped = env.datasets.load_dataframe(next(iter(record.dataset_ids.values())))
    assert not scoped["conditions"].str.contains("999").any()
    assert set(scoped["review_status"]) == {"pending"}
    replay = run(env, finished)
    assert replay.model_call_count == 0 and replay.tool_call_count == 0


def test_normal_factory_shares_service_for_gate_and_report(tmp_path):
    from materials_screening.master.artifact_registry import ArtifactRegistry
    from materials_screening.sub_agents.literature.fulltext_factory import (
        create_fulltext_preview_processor,
    )

    service = object()
    processor = create_fulltext_preview_processor(
        ArtifactRegistry(tmp_path / "artifacts"),
        enable_remote=True,
        analysis_agent_factory=lambda: None,
        figure_review_service=service,
    )
    assert processor.extraction_processor.figure_review_service is service
    assert (
        processor.extraction_processor.analysis_processor.figure_review_service
        is service
    )
