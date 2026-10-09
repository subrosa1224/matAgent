"""Task-bound text review. Callers must hold their server conversation lock."""

import hashlib
import json
import re
from datetime import UTC, datetime

from .figure_evidence_contracts import (
    FigureCandidate,
    FigureEvidenceBatch,
    FigurePageView,
    OcrResult,
    PageRegion,
)
from .figure_evidence_render import render_page, render_source, screen_region
from .figure_evidence_store import batch_reference


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


class FigureEvidenceService:
    def __init__(self, store, artifacts, snapshots, *, ocr=None):
        self.store, self.artifacts, self.snapshots, self.ocr = (
            store,
            artifacts,
            snapshots,
            ocr,
        )

    @staticmethod
    def _revision_id(batch, current, operation_id):
        return (
            "figure-batch-"
            + _digest(
                dict(
                    conversation_id=batch.conversation_id,
                    task_id=batch.task_id,
                    parent=current.model_dump(mode="json"),
                    operation_id=operation_id,
                )
            )[:32]
        )

    @staticmethod
    def _identity_scope(task, conversation_id):
        if task.conversation_id != conversation_id:
            raise ValueError("Figure conversation mismatch")
        if task.figure_review_policy != "figure-evidence-review-v1":
            raise ValueError("Task is not awaiting figure review")
        return {
            doc: ref
            for doc, ref in task.extraction_snapshots.items()
            if task.preview_decisions.get(doc) is not None
            and task.preview_decisions[doc].action == "extract"
        }

    def _pdf(self, task, doc):
        if doc not in self._identity_scope(task, task.conversation_id):
            raise ValueError("Figure document outside current task")
        artifact = "artifact-pdf-" + doc.removeprefix("doc-")
        if artifact not in task.artifact_refs:
            raise ValueError("Figure attachment outside task")
        path = self.artifacts.resolve_pdf(
            artifact, conversation_id=task.conversation_id
        )
        snapshot = self.snapshots.load(task.extraction_snapshots[doc])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if (
            snapshot.status != "complete"
            or snapshot.document_id != doc
            or snapshot.task_id != task.task_id
            or snapshot.conversation_id != task.conversation_id
            or snapshot.pdf_sha256 != digest
        ):
            raise ValueError("Figure snapshot source mismatch")
        return path, digest

    def _scope(self, task, conversation_id):
        scope = self._identity_scope(task, conversation_id)
        if task.stage != "awaiting_figure_review":
            raise ValueError("Task is not awaiting figure review")
        return scope

    def start(self, task, *, conversation_id):
        scope = self._scope(task, conversation_id)
        if task.figure_evidence_ref is not None:
            batch = self._load(task, conversation_id)
            for candidate in batch.candidates:
                if candidate.text_review_status != "skipped":
                    self._candidate_sources(task, candidate)
            return task.figure_evidence_ref
        for doc in scope:
            self._pdf(task, doc)
        return self.store.save(
            FigureEvidenceBatch(
                conversation_id=conversation_id,
                task_id=task.task_id,
                original_question_sha256=_digest(task.original_question),
                selected_snapshots=scope,
            )
        )

    def _load(self, task, conversation_id):
        scope = self._scope(task, conversation_id)
        if task.figure_evidence_ref is None:
            raise ValueError("Missing figure batch")
        batch = self.store.load(task.figure_evidence_ref)
        if (
            batch.conversation_id != conversation_id
            or batch.task_id != task.task_id
            or batch.selected_snapshots != scope
            or batch.original_question_sha256 != _digest(task.original_question)
        ):
            raise ValueError("Figure batch does not match current task")
        return batch

    def report_batch(self, task, ref):
        scope = self._identity_scope(task, task.conversation_id)
        batch = self.store.load(ref)
        if (
            ref != task.figure_evidence_ref
            or batch.status != "closed"
            or batch.task_id != task.task_id
            or batch.conversation_id != task.conversation_id
            or batch.selected_snapshots != scope
            or batch.original_question_sha256 != _digest(task.original_question)
        ):
            raise ValueError("Figure report source scope differs")
        for candidate in batch.candidates:
            if candidate.text_review_status == "text_verified":
                self._candidate_sources(task, candidate)
        return batch

    def select_page(
        self, task, *, conversation_id, expected, operation_id, document_id, page
    ):
        batch, digest, replay = self._operation(
            task,
            conversation_id,
            expected,
            operation_id,
            dict(action="page", document_id=document_id, page=page),
        )
        path, pdf_digest = self._pdf(task, document_id)
        existing = next(
            (
                v
                for v in batch.page_views
                if v.document_id == document_id and v.page == page
            ),
            None,
        )
        if existing is not None:
            if existing.pdf_sha256 != pdf_digest:
                raise ValueError("Source page PDF identity differs")
            self.store.image_path(existing.image)
        if replay:
            if existing is None:
                raise ValueError("Saved page selection is missing")
            return batch_reference(batch)
        if existing is None:
            view = FigurePageView(
                document_id=document_id,
                pdf_sha256=pdf_digest,
                snapshot_ref=task.extraction_snapshots[document_id],
                page=page,
                **render_page(path, page=page, store=self.store),
            )
            views = (*batch.page_views, view)
        else:
            views = batch.page_views
        return self._save(
            batch, task.figure_evidence_ref, operation_id, digest, page_views=views
        )

    def region_from_view(
        self, task, *, conversation_id, document_id, page, image_sha256, region
    ):
        batch = self._load(task, conversation_id)
        _, pdf_digest = self._pdf(task, document_id)
        view = next(
            (
                v
                for v in batch.page_views
                if v.document_id == document_id and v.page == page
            ),
            None,
        )
        if (
            view is None
            or view.pdf_sha256 != pdf_digest
            or view.image.sha256 != image_sha256
        ):
            raise ValueError("Screen selection is not on the current source page")
        self.store.image_path(view.image)
        return screen_region(
            region,
            display_width=view.image.width,
            display_height=view.image.height,
            page_width=view.page_width,
            page_height=view.page_height,
        )

    def _operation(self, task, conversation_id, expected, operation_id, payload):
        batch = self._load(task, conversation_id)
        if (
            not isinstance(operation_id, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}", operation_id) is None
        ):
            raise ValueError("Invalid figure operation ID")
        digest = _digest(payload)
        prior = batch.applied_operations.get(operation_id)
        if prior is not None:
            if prior != digest:
                raise ValueError("Figure operation ID reused for different content")
            return batch, digest, True
        if expected != task.figure_evidence_ref:
            raise ValueError("stale figure revision")
        # The immutable evidence write and checkpoint head write are separate.
        # Recover the one operation-bound child if interrupted between them;
        # never rerender/OCR or accept a changed payload for the same operation.
        recovered = self.store.find_revision(
            self._revision_id(batch, task.figure_evidence_ref, operation_id)
        )
        if recovered is not None:
            if recovered.applied_operations.get(operation_id) != digest:
                raise ValueError("Figure operation ID reused for different content")
            if (
                recovered.parent_ref != task.figure_evidence_ref
                or recovered.revision_number != batch.revision_number + 1
                or recovered.applied_operations
                != {**batch.applied_operations, operation_id: digest}
            ):
                raise ValueError("Interrupted figure revision scope differs")
            self._load(
                task.model_copy(
                    update={"figure_evidence_ref": batch_reference(recovered)}
                ),
                conversation_id,
            )
            return recovered, digest, True
        if batch.status == "closed":
            raise ValueError("Figure batch is closed")
        if batch.revision_number >= 100:
            raise ValueError("Figure revision budget exceeded")
        if batch.revision_number >= 99 and payload["action"] != "close":
            raise ValueError("Last figure revision reserved for explicit closing")
        return batch, digest, False

    def _save(self, batch, current, operation_id, digest, **updates):
        data = dict(
            batch.model_dump(),
            record_id=self._revision_id(batch, current, operation_id),
            parent_ref=current,
            created_at=datetime.now(UTC),
            revision_number=batch.revision_number + 1,
            applied_operations={**batch.applied_operations, operation_id: digest},
            **updates,
        )
        return self.store.save(FigureEvidenceBatch.model_validate(data))

    def _candidate_sources(self, task, candidate):
        _, digest = self._pdf(task, candidate.document_id)
        if (
            candidate.pdf_sha256 != digest
            or candidate.snapshot_ref
            != task.extraction_snapshots[candidate.document_id]
        ):
            raise ValueError("Candidate source changed")
        self.store.image_path(candidate.page_image)
        self.store.image_path(candidate.crop_image)

    def add(
        self,
        task,
        *,
        conversation_id,
        expected,
        operation_id,
        document_id,
        page,
        region,
        figure_label,
        kind,
        replace_candidate_id=None,
        cancel_event=None,
    ):
        region = PageRegion.model_validate(region.model_dump())
        payload = dict(
            action="add",
            doc=document_id,
            page=page,
            region=region.model_dump(),
            figure_label=figure_label,
            kind=kind,
            replace_candidate_id=replace_candidate_id,
        )
        batch, digest, replay = self._operation(
            task, conversation_id, expected, operation_id, payload
        )
        if replay:
            self._pdf(task, document_id)
            for candidate in batch.candidates:
                if (
                    candidate.document_id == document_id
                    and candidate.text_review_status != "skipped"
                ):
                    self._candidate_sources(task, candidate)
            return batch_reference(batch)
        if (
            len([c for c in batch.candidates if c.superseded_by is None]) >= 20
            and replace_candidate_id is None
        ):
            raise ValueError("Active figure budget exceeded")
        replacement = next(
            (
                c
                for c in batch.candidates
                if c.candidate_id == replace_candidate_id and c.superseded_by is None
            ),
            None,
        )
        if replace_candidate_id is not None and replacement is None:
            raise ValueError("Replacement candidate not in current batch")
        # Validate client text/kind before rendering or consuming local OCR.
        if (
            not isinstance(figure_label, str)
            or not 1 <= len(figure_label) <= 160
            or kind not in {"axis_label", "axis_ticks", "legend_text", "other_text"}
        ):
            raise ValueError("Invalid figure label or text kind")
        if cancel_event is not None and cancel_event.is_set():
            raise ValueError("Figure rendering cancelled")
        path, pdf_digest = self._pdf(task, document_id)
        source = render_source(
            path, page=page, region=region, store=self.store, cancel_event=cancel_event
        )
        if cancel_event is not None and cancel_event.is_set():
            raise ValueError("Figure OCR cancelled")
        result = OcrResult(status="unavailable", error_code="NO_LOCAL_OCR")
        if self.ocr is not None:
            try:
                image_path = self.store.image_path(source["crop_image"])
                recognized = (
                    self.ocr.recognize(image_path, cancel_event=cancel_event)
                    if hasattr(self.ocr, "recognize")
                    else self.ocr(image_path)
                )
                result = OcrResult.model_validate(recognized.model_dump())
            except Exception:
                result = OcrResult(status="failed", error_code="LOCAL_OCR_FAILED")
        if cancel_event is not None and cancel_event.is_set():
            raise ValueError("Figure OCR cancelled")
        candidate = FigureCandidate(
            document_id=document_id,
            artifact_id="artifact-pdf-" + document_id.removeprefix("doc-"),
            pdf_sha256=pdf_digest,
            snapshot_ref=task.extraction_snapshots[document_id],
            page=page,
            region=region,
            figure_label=figure_label,
            kind=kind,
            ocr=result,
            **source,
        )
        self._candidate_sources(task, candidate)
        candidates = tuple(
            FigureCandidate.model_validate(
                {
                    **c.model_dump(),
                    "superseded_by": candidate.candidate_id,
                    "text_review_status": "skipped",
                }
            )
            if c == replacement
            else c
            for c in batch.candidates
        )
        return self._save(
            batch,
            task.figure_evidence_ref,
            operation_id,
            digest,
            candidates=(*candidates, candidate),
        )

    def decide(
        self,
        task,
        *,
        conversation_id,
        expected,
        operation_id,
        candidate_id,
        action,
        text=None,
    ):
        if action not in {"text_verified", "skipped"}:
            raise ValueError("Only explicit text verification or skipping is allowed")
        payload = dict(action=action, candidate=candidate_id, text=text)
        batch, digest, replay = self._operation(
            task, conversation_id, expected, operation_id, payload
        )
        if replay:
            if action == "text_verified":
                candidate = next(
                    (c for c in batch.candidates if c.candidate_id == candidate_id),
                    None,
                )
                if candidate is None:
                    raise ValueError("Unknown replay candidate")
                self._candidate_sources(task, candidate)
            return batch_reference(batch)
        candidate = next(
            (
                c
                for c in batch.candidates
                if c.candidate_id == candidate_id and c.superseded_by is None
            ),
            None,
        )
        if candidate is None:
            raise ValueError("Unknown or superseded figure candidate")
        if action == "text_verified":
            self._candidate_sources(task, candidate)
            if not isinstance(text, str) or not text.strip() or len(text) > 10000:
                raise ValueError(
                    "Text verification requires bounded nonempty transcription"
                )
        elif text is not None:
            raise ValueError("Skip does not submit reviewed text")
        updated = FigureCandidate.model_validate(
            dict(
                candidate.model_dump(),
                reviewed_text=text,
                text_review_status=action,
                reviewer=conversation_id,
                reviewed_at=datetime.now(UTC),
                manual_transcription=action == "text_verified"
                and candidate.ocr.status != "ok",
            )
        )
        return self._save(
            batch,
            task.figure_evidence_ref,
            operation_id,
            digest,
            candidates=tuple(
                updated if c.candidate_id == candidate_id else c
                for c in batch.candidates
            ),
        )

    def close(
        self,
        task,
        *,
        conversation_id,
        expected,
        operation_id,
        skip_remaining=False,
        abandon_all=False,
    ):
        if type(skip_remaining) is not bool or type(abandon_all) is not bool:
            raise ValueError("Explicit skip flag must be boolean")
        batch, digest, replay = self._operation(
            task,
            conversation_id,
            expected,
            operation_id,
            dict(
                action="close", skip_remaining=skip_remaining, abandon_all=abandon_all
            ),
        )
        if replay:
            for candidate in batch.candidates:
                if candidate.text_review_status == "text_verified":
                    self._candidate_sources(task, candidate)
            return batch_reference(batch)
        if not (skip_remaining or abandon_all) and any(
            c.text_review_status == "pending" for c in batch.candidates
        ):
            raise ValueError("pending figures require an explicit decision")
        candidates = []
        for c in batch.candidates:
            if c.text_review_status == "text_verified" and not abandon_all:
                self._candidate_sources(task, c)
            if c.text_review_status == "pending" or abandon_all:
                c = FigureCandidate.model_validate(
                    dict(
                        c.model_dump(),
                        text_review_status="skipped",
                        reviewed_text=None,
                        manual_transcription=False,
                        reviewer=conversation_id,
                        reviewed_at=datetime.now(UTC),
                    )
                )
            candidates.append(c)
        return self._save(
            batch,
            task.figure_evidence_ref,
            operation_id,
            digest,
            candidates=tuple(candidates),
            status="closed",
        )
