"""Local control tests; stub snapshots/OCR never stand for a real Agent run."""

import hashlib
import threading
from datetime import timedelta
from types import SimpleNamespace

import pymupdf
import pytest
from pydantic import ValidationError

from materials_screening.master.artifact_registry import ArtifactRegistry
from materials_screening.master.figure_evidence_contracts import OcrResult, PageRegion
from materials_screening.master.figure_evidence_review import FigureEvidenceService
from materials_screening.master.figure_evidence_store import FigureEvidenceStore
from materials_screening.master.fulltext_snapshots import SnapshotReference
from materials_screening.master.fulltext_tasks import FulltextTask, PreviewDecision
from materials_screening.sub_agents.literature.preview import PaperPreview


@pytest.fixture
def setup(tmp_path):
    pdf = tmp_path / "fixture.pdf"
    with pymupdf.open() as document:
        page = document.new_page(width=300, height=400)
        page.insert_text((30, 100), "Irradiation time (min) 0 20 180")
        document.save(pdf)
    registry = ArtifactRegistry(tmp_path / "artifacts")
    artifact = registry.register_pdf(pdf, conversation_id="unit-figure")
    digest = hashlib.sha256(pdf.read_bytes()).hexdigest()
    doc = "doc-" + digest[:24]
    ref = SnapshotReference(
        snapshot_id="snapshot-" + "a" * 32, document_id=doc, content_sha256="b" * 64
    )
    # Minimal server task with real identity; preview omitted only by constructing
    # a preview through the existing fixture parser when required below.
    task = FulltextTask(
        conversation_id="unit-figure",
        original_question="提取光催化实验",
        artifact_refs=(artifact.artifact_id,),
        document_ids=(doc,),
        extraction_snapshots={doc: ref},
        snapshot_history=(ref,),
        stage="awaiting_figure_review",
        figure_review_policy="figure-evidence-review-v1",
    )
    preview = PaperPreview(
        document_id=doc,
        topic=task.original_question,
        title="unit",
        article_type="research",
        topic_relevance="core",
        research_question="unit",
        methods="unit",
        key_findings="unit",
        recommendation="deep_analyze",
        reason="unit",
        evidence_quote="unit",
        chunk_id="unit",
        page_from=1,
        page_to=1,
        source_text_sha256="c" * 64,
        evidence_quality="verbatim",
    )
    task = task.model_copy(
        update={
            "previews": (preview,),
            "preview_decisions": {
                doc: PreviewDecision(action="extract", reason="unit scope")
            },
        }
    )
    # Service only needs server-selected decisions, not generated preview fields.
    snapshot = SimpleNamespace(
        task_id=task.task_id,
        conversation_id=task.conversation_id,
        document_id=doc,
        pdf_sha256=digest,
        status="complete",
    )
    snapshots = SimpleNamespace(load=lambda reference: snapshot)
    store = FigureEvidenceStore(tmp_path / "figures")
    calls = []

    def ocr(path):
        calls.append(path)
        return OcrResult(
            status="ok",
            engine="unit-stub",
            language="en-unit",
            raw_text="1 80 lrradiation",
        )

    service = FigureEvidenceService(store, registry, snapshots, ocr=ocr)
    head = service.start(task, conversation_id="unit-figure")
    task = task.model_copy(update={"figure_evidence_ref": head})
    return SimpleNamespace(
        task=task, doc=doc, service=service, store=store, calls=calls, registry=registry
    )


def add(s, operation="add-1", label="unit-figure", **kw):
    return s.service.add(
        s.task,
        conversation_id=s.task.conversation_id,
        expected=s.task.figure_evidence_ref,
        operation_id=operation,
        document_id=s.doc,
        page=1,
        region=PageRegion(x0=20, y0=70, x1=280, y1=120),
        figure_label=label,
        kind="axis_label",
        **kw,
    )


def update(s, ref):
    s.task = s.task.model_copy(update={"figure_evidence_ref": ref})


@pytest.mark.parametrize(
    "coords",
    [
        (0, 0, 0, 10),
        (20, 0, 10, 10),
        (-1, 0, 10, 10),
        (0, 0, float("nan"), 10),
        (0, 0, float("inf"), 10),
    ],
)
def test_bad_rectangles_fail(coords):
    with pytest.raises(ValidationError):
        PageRegion(x0=coords[0], y0=coords[1], x1=coords[2], y1=coords[3])


def test_confirmation_is_text_only_and_preserves_original(setup):
    s = setup
    before = s.task.model_dump(mode="json", exclude={"previews"})
    first = add(s)
    raw = s.store.load(first)
    candidate = raw.candidates[0]
    update(s, first)
    decided = s.service.decide(
        s.task,
        conversation_id=s.task.conversation_id,
        expected=first,
        operation_id="confirm-1",
        candidate_id=candidate.candidate_id,
        action="text_verified",
        text="180 Irradiation time (min)",
    )
    verified = s.store.load(decided).candidates[0]
    assert verified.ocr.raw_text == "1 80 lrradiation"
    assert verified.reviewed_text == "180 Irradiation time (min)"
    assert verified.condition_binding_status == "unbound"
    assert verified.text_review_status == "text_verified"
    assert s.store.load(first) == raw
    assert s.task.model_dump(
        mode="json", exclude={"previews", "figure_evidence_ref"}
    ) == {k: v for k, v in before.items() if k != "figure_evidence_ref"}
    assert verified.reviewer == s.task.conversation_id


def test_duplicate_action_does_not_repeat_ocr_and_stale_action_rejected(setup):
    s = setup
    first = add(s)
    update(s, first)
    assert add(s) == first
    assert len(s.calls) == 1
    with pytest.raises(ValueError, match="stale"):
        s.service.close(
            s.task,
            conversation_id=s.task.conversation_id,
            expected=first.model_copy(update={"content_sha256": "0" * 64}),
            operation_id="close-1",
        )
    with pytest.raises(ValueError, match="operation"):
        add(s, label="different")


def test_pending_requires_explicit_skip_and_close_is_idempotent(setup):
    s = setup
    update(s, add(s))
    with pytest.raises(ValueError, match="pending"):
        s.service.close(
            s.task,
            conversation_id=s.task.conversation_id,
            expected=s.task.figure_evidence_ref,
            operation_id="close-1",
        )
    closed = s.service.close(
        s.task,
        conversation_id=s.task.conversation_id,
        expected=s.task.figure_evidence_ref,
        operation_id="close-1",
        skip_remaining=True,
    )
    update(s, closed)
    assert s.store.load(closed).status == "closed"
    assert s.store.load(closed).candidates[0].text_review_status == "skipped"
    assert (
        s.service.close(
            s.task,
            conversation_id=s.task.conversation_id,
            expected=closed,
            operation_id="close-1",
            skip_remaining=True,
        )
        == closed
    )
    with pytest.raises(ValueError, match="closed"):
        add(s, operation="new-after-close")


def test_cross_conversation_and_source_changes_rejected(setup):
    s = setup
    with pytest.raises(ValueError, match="conversation"):
        s.service.start(s.task, conversation_id="other")
    with pytest.raises(ValueError):
        s.service.add(
            s.task,
            conversation_id=s.task.conversation_id,
            expected=s.task.figure_evidence_ref,
            operation_id="bad-doc",
            document_id="doc-" + "c" * 24,
            page=1,
            region=PageRegion(x0=1, y0=1, x1=20, y1=20),
            figure_label="bad",
            kind="legend_text",
        )
    path = s.registry.resolve_pdf(
        s.task.artifact_refs[0], conversation_id=s.task.conversation_id
    )
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError):
        add(s)


def test_tampered_images_cannot_confirm_but_can_be_explicitly_skipped(setup):
    s = setup
    first = add(s)
    update(s, first)
    candidate = s.store.load(first).candidates[0]
    path = s.store.image_path(candidate.crop_image)
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError):
        s.service.decide(
            s.task,
            conversation_id=s.task.conversation_id,
            expected=first,
            operation_id="confirm",
            candidate_id=candidate.candidate_id,
            action="text_verified",
            text="180",
        )
    skipped = s.service.decide(
        s.task,
        conversation_id=s.task.conversation_id,
        expected=first,
        operation_id="skip",
        candidate_id=candidate.candidate_id,
        action="skipped",
    )
    update(s, skipped)
    assert (
        s.store.load(
            s.service.close(
                s.task,
                conversation_id=s.task.conversation_id,
                expected=skipped,
                operation_id="close",
            )
        ).status
        == "closed"
    )


def test_unavailable_ocr_manual_transcription_and_restart(setup):
    s = setup
    s.service.ocr = None
    first = add(s)
    update(s, first)
    candidate = s.store.load(first).candidates[0]
    assert candidate.ocr.status == "unavailable"
    verified = s.service.decide(
        s.task,
        conversation_id=s.task.conversation_id,
        expected=first,
        operation_id="manual",
        candidate_id=candidate.candidate_id,
        action="text_verified",
        text="manual axis",
    )
    assert s.store.load(verified).candidates[0].manual_transcription
    reloaded = FigureEvidenceStore(s.store.root).load(verified)
    assert reloaded == s.store.load(verified)


def test_page_region_resource_limits_and_immutable_record(setup):
    s = setup
    for page, rect in [
        (2, PageRegion(x0=1, y0=1, x1=20, y1=20)),
        (1, PageRegion(x0=1, y0=1, x1=301, y1=20)),
    ]:
        with pytest.raises(ValueError):
            s.service.add(
                s.task,
                conversation_id=s.task.conversation_id,
                expected=s.task.figure_evidence_ref,
                operation_id="bad-crop",
                document_id=s.doc,
                page=page,
                region=rect,
                figure_label="x",
                kind="other_text",
            )
    first = add(s)
    batch = s.store.load(first)
    assert s.store.save(batch) == first
    with pytest.raises(ValueError, match="overwrite"):
        s.store.save(
            batch.model_copy(
                update={"created_at": batch.created_at + timedelta(seconds=1)}
            )
        )
    with pytest.raises(ValueError, match="pending"):
        s.store.save(batch.model_copy(update={"status": "closed"}))


def test_old_tasks_default_disabled():
    task = FulltextTask(conversation_id="old", original_question="old task")
    assert task.figure_review_policy == "disabled"
    assert task.figure_evidence_ref is None


def test_cancel_stops_before_ocr(setup):
    event = threading.Event()
    event.set()
    with pytest.raises(ValueError, match="cancelled"):
        add(setup, cancel_event=event)
    assert not setup.calls


def test_wrong_snapshot_context_is_rejected(setup):
    s = setup
    s.service.snapshots.load(s.task.extraction_snapshots[s.doc]).task_id = (
        "task-fulltext-" + "d" * 32
    )
    with pytest.raises(ValueError, match="snapshot"):
        add(s)


def test_source_change_during_ocr_never_commits_candidate(setup):
    s = setup
    head = s.task.figure_evidence_ref
    path = s.registry.resolve_pdf(
        s.task.artifact_refs[0], conversation_id=s.task.conversation_id
    )

    def changing_ocr(image):
        path.write_bytes(path.read_bytes() + b"changed")
        return OcrResult(status="ok", raw_text="180", engine="unit")

    s.service.ocr = changing_ocr
    with pytest.raises(ValueError):
        add(s)
    assert s.store.load(head).candidates == ()


def test_confirm_replay_rechecks_image_and_abandon_all_exits(setup):
    s = setup
    update(s, add(s))
    c = s.store.load(s.task.figure_evidence_ref).candidates[0]
    args = dict(
        conversation_id=s.task.conversation_id,
        operation_id="verify",
        candidate_id=c.candidate_id,
        action="text_verified",
        text="180",
    )
    update(s, s.service.decide(s.task, expected=s.task.figure_evidence_ref, **args))
    path = s.store.image_path(c.crop_image)
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError):
        s.service.decide(s.task, expected=s.task.figure_evidence_ref, **args)
    closed = s.service.close(
        s.task,
        conversation_id=s.task.conversation_id,
        expected=s.task.figure_evidence_ref,
        operation_id="abandon",
        abandon_all=True,
    )
    assert s.store.load(closed).candidates[0].text_review_status == "skipped"
    assert s.store.load(closed).candidates[0].reviewed_text is None


def test_last_revision_reserved_for_explicit_close(setup):
    s = setup
    update(s, add(s))
    batch = s.store.load(s.task.figure_evidence_ref)
    from uuid import uuid4

    limit = batch.model_copy(
        update={"record_id": "figure-batch-" + uuid4().hex, "revision_number": 99}
    )
    update(s, s.store.save(limit))
    with pytest.raises(ValueError, match="reserved"):
        add(s, operation="too-late")
    closed = s.service.close(
        s.task,
        conversation_id=s.task.conversation_id,
        expected=s.task.figure_evidence_ref,
        operation_id="last-close",
        abandon_all=True,
    )
    assert s.store.load(closed).revision_number == 100


def test_ocr_exception_is_safe_and_manual_review_available(setup):
    s = setup

    def bad_ocr(path):
        raise RuntimeError("DO NOT EXPOSE private details")

    s.service.ocr = bad_ocr
    ref = add(s)
    candidate = s.store.load(ref).candidates[0]
    assert candidate.ocr.status == "failed"
    assert candidate.ocr.error_code == "LOCAL_OCR_FAILED"
    assert "private" not in s.store.load(ref).model_dump_json()


def test_no_extra_condition_fields_allowed(setup):
    c = setup.store.load(add(setup)).candidates[0]
    with pytest.raises(ValidationError):
        type(c).model_validate({**c.model_dump(), "irradiation_time": 180})
    with pytest.raises(ValidationError):
        type(c).model_validate({**c.model_dump(), "condition_binding_status": "bound"})


def test_replacement_preserves_old_region_and_candidate(setup):
    s = setup
    old = add(s)
    update(s, old)
    candidate = s.store.load(old).candidates[0]
    replaced = add(
        s, operation="replacement", replace_candidate_id=candidate.candidate_id
    )
    new = s.store.load(replaced)
    assert len(new.candidates) == 2
    assert new.candidates[0].superseded_by == new.candidates[1].candidate_id
    assert new.candidates[0].text_review_status == "skipped"
    assert s.store.load(old).candidates[0].superseded_by is None


def test_screen_region_maps_to_unrotated_page():
    from materials_screening.master.figure_evidence_render import screen_region

    rect = screen_region(
        PageRegion(x0=10, y0=20, x1=100, y1=120),
        display_width=200,
        display_height=400,
        page_width=100,
        page_height=200,
    )
    assert rect == PageRegion(x0=5, y0=10, x1=50, y1=60)
    with pytest.raises(ValueError):
        screen_region(
            rect, display_width=0, display_height=400, page_width=100, page_height=200
        )


def test_rotated_pdf_is_rendered_unrotated_without_rewriting(tmp_path):
    from materials_screening.master.figure_evidence_render import render_source

    path = tmp_path / "rotated.pdf"
    with pymupdf.open() as document:
        page = document.new_page(width=300, height=400)
        page.set_rotation(90)
        document.save(path)
    original = path.read_bytes()
    store = FigureEvidenceStore(tmp_path / "images")
    result = render_source(
        path, page=1, region=PageRegion(x0=0, y0=0, x1=50, y1=50), store=store
    )
    assert result["original_rotation"] == 90
    assert result["page_width"] == 300 and result["page_height"] == 400
    assert result["page_image"].width < result["page_image"].height
    assert path.read_bytes() == original


def test_oversized_page_rejected_before_render(tmp_path):
    from materials_screening.master.figure_evidence_render import render_source

    path = tmp_path / "huge.pdf"
    with pymupdf.open() as document:
        document.new_page(width=10000, height=10000)
        document.save(path)
    with pytest.raises(ValueError, match="pixel"):
        render_source(
            path,
            page=1,
            region=PageRegion(x0=0, y0=0, x1=50, y1=50),
            store=FigureEvidenceStore(tmp_path / "images"),
        )


def test_overlong_review_not_written(setup):
    s = setup
    head = add(s)
    update(s, head)
    candidate = s.store.load(head).candidates[0]
    with pytest.raises(ValueError, match="bounded"):
        s.service.decide(
            s.task,
            conversation_id=s.task.conversation_id,
            expected=head,
            operation_id="too-long",
            candidate_id=candidate.candidate_id,
            action="text_verified",
            text="x" * 10001,
        )
    assert s.store.load(head).candidates[0].text_review_status == "pending"


def test_gate_pauses_before_analysis_and_reuses_closed_review(setup):
    from materials_screening.master.figure_evidence_gate import FigureReviewGate

    s = setup
    task = s.task.model_copy(
        update={"stage": "extracting", "figure_evidence_ref": None}
    )
    saved = []
    gate = FigureReviewGate(s.service)
    paused, result = gate.before_analysis(
        task, save_task=saved.append, user_turn_id="unit", model_calls=2
    )
    assert result.final_status == "needs_user_input"
    assert paused.stage == "awaiting_figure_review"
    assert result.model_call_count == 2 and not s.calls
    head = s.service.close(
        paused,
        conversation_id=paused.conversation_id,
        expected=paused.figure_evidence_ref,
        operation_id="skip",
        abandon_all=True,
    )
    paused = paused.model_copy(update={"figure_evidence_ref": head})
    ready, result = gate.before_analysis(
        paused, save_task=saved.append, user_turn_id="unit", model_calls=2
    )
    assert result is None
    assert ready.figure_evidence_ref == head


def test_gate_disabled_preserves_legacy_task(setup):
    from materials_screening.master.figure_evidence_gate import FigureReviewGate

    s = setup
    task = s.task.model_copy(
        update={
            "stage": "extracting",
            "figure_review_policy": "disabled",
            "figure_evidence_ref": None,
        }
    )
    same, result = FigureReviewGate(s.service).before_analysis(
        task,
        save_task=lambda t: pytest.fail("must not save"),
        user_turn_id="unit",
        model_calls=0,
    )
    assert same is task and result is None


def test_report_appendix_is_escaped_and_never_claims_condition_binding(setup):
    from materials_screening.master.figure_evidence_gate import render_figure_appendix

    s = setup
    head = add(s)
    update(s, head)
    c = s.store.load(head).candidates[0]
    ref = s.service.decide(
        s.task,
        conversation_id=s.task.conversation_id,
        expected=head,
        operation_id="verify",
        candidate_id=c.candidate_id,
        action="text_verified",
        text="<script>[click](https://evil.example)</script>",
    )
    text = render_figure_appendix(s.store.load(ref))
    assert "<script>" not in text and "[click](https://evil.example)" not in text
    assert "实验条件尚未绑定" in text


def test_master_structured_action_uses_server_task_and_lock(setup):
    import threading

    from materials_screening.master.master_runner import MasterAgentRunner

    s = setup
    runner = object.__new__(MasterAgentRunner)
    runner._figure_review_service = s.service
    lock = threading.Lock()
    runner._store = SimpleNamespace(conversation_lock=lambda c: lock)
    runner.get_fulltext_tasks = lambda c: (s.task,)
    runner._save_fulltext_task = lambda t: update(s, t.figure_evidence_ref)
    batch = runner.figure_review_action(
        conversation_id=s.task.conversation_id,
        task_id=s.task.task_id,
        expected=s.task.figure_evidence_ref,
        operation_id="close",
        action="close",
        abandon_all=True,
    )
    assert batch.status == "closed"
    lock.acquire()
    try:
        with pytest.raises(ValueError, match="busy"):
            runner.figure_review_action(
                conversation_id=s.task.conversation_id,
                task_id=s.task.task_id,
                expected=s.task.figure_evidence_ref,
                operation_id="later",
                action="close",
            )
    finally:
        lock.release()


def test_page_view_is_persisted_and_screen_mapping_is_server_owned(setup):
    s = setup
    ref = s.service.select_page(
        s.task,
        conversation_id=s.task.conversation_id,
        expected=s.task.figure_evidence_ref,
        operation_id="page",
        document_id=s.doc,
        page=1,
    )
    update(s, ref)
    view = s.store.load(ref).page_views[0]
    assert (
        s.service.select_page(
            s.task,
            conversation_id=s.task.conversation_id,
            expected=ref,
            operation_id="page",
            document_id=s.doc,
            page=1,
        )
        == ref
    )
    mapped = s.service.region_from_view(
        s.task,
        conversation_id=s.task.conversation_id,
        document_id=s.doc,
        page=1,
        image_sha256=view.image.sha256,
        region=PageRegion(
            x0=0, y0=0, x1=view.image.width / 2, y1=view.image.height / 2
        ),
    )
    assert mapped == PageRegion(x0=0, y0=0, x1=150, y1=200)
    with pytest.raises(ValueError):
        s.service.region_from_view(
            s.task,
            conversation_id=s.task.conversation_id,
            document_id=s.doc,
            page=1,
            image_sha256="a" * 64,
            region=PageRegion(x0=0, y0=0, x1=20, y1=20),
        )


def test_saved_evidence_replays_after_task_head_write_interruption(setup):
    s = setup
    parent = s.task.figure_evidence_ref
    ref = add(s)
    # Simulate evidence committed, process interrupted before task head commit.
    assert s.task.figure_evidence_ref == parent
    assert add(s) == ref and len(s.calls) == 1
    update(s, ref)
    candidate = s.store.load(ref).candidates[0]
    args = dict(
        conversation_id=s.task.conversation_id,
        expected=ref,
        operation_id="interrupted-confirm",
        candidate_id=candidate.candidate_id,
        action="text_verified",
        text="180 min",
    )
    confirmed = s.service.decide(s.task, **args)
    assert s.task.figure_evidence_ref == ref
    assert s.service.decide(s.task, **args) == confirmed
    update(s, confirmed)
    args = dict(
        conversation_id=s.task.conversation_id,
        expected=confirmed,
        operation_id="interrupted-close",
    )
    closed = s.service.close(s.task, **args)
    assert s.service.close(s.task, **args) == closed
    assert s.store.load(closed).status == "closed"


def test_interrupted_operation_cannot_be_reused_with_changed_content(setup):
    s = setup
    ref = add(s)
    assert s.task.figure_evidence_ref != ref
    with pytest.raises(ValueError, match="different content"):
        add(s, label="different")
    assert len(s.calls) == 1
